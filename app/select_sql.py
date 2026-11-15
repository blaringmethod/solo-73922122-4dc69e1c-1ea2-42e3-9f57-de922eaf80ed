"""单条 SELECT 的静态解析（只解析，绝不执行）。

支持的语法子集：
* 单条 SELECT，可选一层非递归 WITH（CTE 可引用先定义的 CTE）；
* FROM 中至多两张源表、表别名、JOIN … ON（也识别逗号连接）；
* 投影：``*``、``别名.*``、列引用、字面量、四则运算（+ - * /）；
* 条件：列比较（= == != <> < <= > >=）与 AND/OR。

其它语法（函数、子查询、DISTINCT、GROUP BY、NOT、IN/LIKE/IS、
UNION、递归/嵌套 WITH、第三张源表等）记录为带位置的 Problem，
交由分析层继续尽量归因；只要存在 Problem，整体结论就不得为“无影响”。
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 词法
# ---------------------------------------------------------------------------


class ParseFatal(Exception):
    """无法继续解析的结构性错误（带位置）。"""

    def __init__(self, pos: int, message: str):
        super().__init__(message)
        self.pos = pos
        self.message = message


class _Unsupported(Exception):
    """可识别但超出支持范围的语法（带位置），记录后尝试恢复继续解析。"""

    def __init__(self, pos: int, message: str):
        super().__init__(message)
        self.pos = pos
        self.message = message


@dataclass
class Problem:
    """解析期间发现的“超出支持范围/无法确认”位置。"""

    pos: int
    message: str


@dataclass
class Token:
    kind: str  # ident / qident / number / string / op / eof
    text: str
    pos: int

    @property
    def upper(self) -> str:
        return self.text.upper()


_KEYWORDS = {
    "WITH", "RECURSIVE", "AS", "SELECT", "DISTINCT", "ALL", "FROM", "WHERE",
    "AND", "OR", "NOT", "JOIN", "INNER", "LEFT", "OUTER", "CROSS", "RIGHT",
    "FULL", "NATURAL", "ON", "USING", "GROUP", "HAVING", "ORDER", "LIMIT",
    "OFFSET", "UNION", "EXCEPT", "INTERSECT", "BY", "ASC", "DESC", "NULL",
    "TRUE", "FALSE", "CASE", "WHEN", "THEN", "ELSE", "END", "IN", "IS",
    "LIKE", "GLOB", "BETWEEN", "EXISTS", "CAST", "COLLATE", "ESCAPE",
    "MATCH", "REGEXP", "ISNULL", "NOTNULL", "WINDOW", "FILTER", "OVER",
    "VALUES",
}

# 恢复跳过时的子句边界关键字
_BOUNDARY_KWS = {
    "FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "OFFSET", "WINDOW",
    "UNION", "EXCEPT", "INTERSECT", "JOIN", "INNER", "LEFT", "RIGHT", "FULL",
    "CROSS", "NATURAL", "ON", "USING", "VALUES",
}

# 不能作为裸列引用出现的关键字（NULL/TRUE/FALSE 单独按字面量处理）
_NOT_REF_KEYWORDS = _KEYWORDS - {"NULL", "TRUE", "FALSE"}

_MULTI_OPS = ("<=", ">=", "<>", "!=", "==", "||", "<<", ">>")
_SINGLE_OPS = set("(),.;+-*/%=<>&|")
_COMPARE_OPS = {"=", "==", "!=", "<>", "<", "<=", ">", ">="}


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                raise ParseFatal(i, "块注释未闭合。")
            i = j + 2
            continue
        if ch == "'":
            j = i + 1
            while j < n:
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                raise ParseFatal(i, "字符串字面量未闭合。")
            tokens.append(Token("string", text[i:j + 1], i))
            i = j + 1
            continue
        if ch in '"`[':
            close = "]" if ch == "[" else ch
            j = i + 1
            buf: list[str] = []
            closed = False
            while j < n:
                if text[j] == close:
                    if close != "]" and j + 1 < n and text[j + 1] == close:
                        buf.append(close)
                        j += 2
                        continue
                    closed = True
                    break
                buf.append(text[j])
                j += 1
            if not closed:
                raise ParseFatal(i, "引号标识符未闭合。")
            tokens.append(Token("qident", "".join(buf), i))
            i = j + 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            if j < n and text[j] in "eE":
                k = j + 1
                if k < n and text[k] in "+-":
                    k += 1
                if k < n and text[k].isdigit():
                    while k < n and text[k].isdigit():
                        k += 1
                    j = k
            tokens.append(Token("number", text[i:j], i))
            i = j
            continue
        if ch.isalpha() or ch == "_" or ord(ch) > 127:
            j = i
            while j < n and (
                text[j].isalnum() or text[j] in "_$" or ord(text[j]) > 127
            ):
                j += 1
            tokens.append(Token("ident", text[i:j], i))
            i = j
            continue
        two = text[i:i + 2]
        if two in _MULTI_OPS:
            tokens.append(Token("op", two, i))
            i += 2
            continue
        if ch in _SINGLE_OPS:
            tokens.append(Token("op", ch, i))
            i += 1
            continue
        raise ParseFatal(i, f"无法识别的字符 {ch!r}。")
    tokens.append(Token("eof", "", n))
    return tokens


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass
class Expr:
    pos: int


@dataclass
class ColumnRef(Expr):
    table: str | None
    name: str


@dataclass
class Literal(Expr):
    text: str


@dataclass
class Unary(Expr):
    op: str
    operand: Expr


@dataclass
class Binary(Expr):  # 四则运算
    op: str
    left: Expr
    right: Expr


@dataclass
class Compare(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class Logic(Expr):  # AND / OR
    op: str
    left: Expr
    right: Expr


@dataclass
class UnknownExpr(Expr):
    """解析失败后占位的表达式：依赖关系一律按“无法确定”处理。"""


@dataclass
class Item:
    pos: int


@dataclass
class StarItem(Item):
    pass


@dataclass
class TableStarItem(Item):
    table: str


@dataclass
class ExprItem(Item):
    expr: Expr
    alias: str | None


@dataclass
class UnknownItem(Item):
    pass


@dataclass
class SourceRef:
    pos: int
    name: str
    alias: str | None


@dataclass
class Join:
    pos: int
    source: SourceRef
    on: Expr | None  # None 表示无连接条件（CROSS/逗号）


@dataclass
class Select:
    pos: int
    items: list[Item]
    sources: list[SourceRef]
    joins: list[Join]
    where: Expr | None


@dataclass
class Cte:
    pos: int
    name: str
    columns: list[str] | None  # AS (x, y) 显式改名清单
    body: "Query"


@dataclass
class Query:
    pos: int
    ctes: list[Cte]
    select: Select


def render_expr(expr: Expr) -> str:
    """把表达式渲染回可读的 SQL 文本（用于报告展示）。"""
    if isinstance(expr, ColumnRef):
        return f"{expr.table}.{expr.name}" if expr.table else expr.name
    if isinstance(expr, Literal):
        return expr.text
    if isinstance(expr, Unary):
        return f"{expr.op}{render_expr(expr.operand)}"
    if isinstance(expr, Binary):
        return f"({render_expr(expr.left)} {expr.op} {render_expr(expr.right)})"
    if isinstance(expr, Compare):
        op = "=" if expr.op == "==" else expr.op
        return f"{render_expr(expr.left)} {op} {render_expr(expr.right)}"
    if isinstance(expr, Logic):
        return f"({render_expr(expr.left)} {expr.op} {render_expr(expr.right)})"
    if isinstance(expr, UnknownExpr):
        return "〈无法解析的部分〉"
    return "〈表达式〉"  # pragma: no cover - 防御性兜底


# ---------------------------------------------------------------------------
# 语法分析
# ---------------------------------------------------------------------------


def parse_query(text: str) -> tuple[Query, list[Problem]]:
    """解析单条 SELECT，返回 (AST, 问题列表)。

    结构性错误抛出 ParseFatal；超出支持范围但可恢复的语法记入问题列表。
    """
    parser = _Parser(text)
    query = parser.parse_single()
    return query, parser.problems


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = tokenize(text)
        self.i = 0
        self.problems: list[Problem] = []

    # -- 词法单元辅助 ------------------------------------------------------

    def _peek(self, k: int = 0) -> Token:
        return self.tokens[min(self.i + k, len(self.tokens) - 1)]

    def _advance(self) -> Token:
        tok = self.tokens[self.i]
        if tok.kind != "eof":
            self.i += 1
        return tok

    def _at_op(self, op: str) -> bool:
        tok = self._peek()
        return tok.kind == "op" and tok.text == op

    def _at_kw(self, kw: str) -> bool:
        tok = self._peek()
        return tok.kind == "ident" and tok.upper == kw

    def _problem(self, pos: int, message: str) -> None:
        self.problems.append(Problem(pos, message))

    @staticmethod
    def _describe(tok: Token) -> str:
        return "语句结束" if tok.kind == "eof" else f"“{tok.text}”"

    def _expect_op(self, op: str) -> Token:
        tok = self._peek()
        if tok.kind == "op" and tok.text == op:
            return self._advance()
        raise ParseFatal(tok.pos, f"此处应为 “{op}”，实际是 {self._describe(tok)}。")

    def _expect_kw(self, kw: str) -> Token:
        tok = self._peek()
        if tok.kind == "ident" and tok.upper == kw:
            return self._advance()
        raise ParseFatal(
            tok.pos, f"此处应为 {kw}，实际是 {self._describe(tok)}。"
        )

    def _expect_ident(self, what: str, allow_string: bool = False) -> Token:
        tok = self._peek()
        if tok.kind in ("ident", "qident"):
            return self._advance()
        if allow_string and tok.kind == "string":
            self._advance()
            return Token("qident", tok.text[1:-1].replace("''", "'"), tok.pos)
        raise ParseFatal(
            tok.pos, f"此处应为{what}，实际是 {self._describe(tok)}。"
        )

    def _at_bare_alias(self) -> bool:
        tok = self._peek()
        if tok.kind in ("qident", "string"):
            return True
        return tok.kind == "ident" and tok.upper not in _KEYWORDS

    # -- 恢复跳过 ----------------------------------------------------------

    def _skip_to_boundary(self, stop_comma: bool = False) -> None:
        depth = 0
        while True:
            tok = self._peek()
            if tok.kind == "eof":
                return
            if tok.kind == "op":
                if tok.text == "(":
                    depth += 1
                elif tok.text == ")":
                    if depth == 0:
                        return
                    depth -= 1
                elif depth == 0 and tok.text == ";":
                    return
                elif depth == 0 and stop_comma and tok.text == ",":
                    return
            elif (
                depth == 0
                and tok.kind == "ident"
                and tok.upper in _BOUNDARY_KWS
            ):
                return
            self._advance()

    def _skip_to_statement_end(self) -> None:
        """跳过复合查询的其余分支：直到语句结束（;、) 或 EOF）。"""
        depth = 0
        while True:
            tok = self._peek()
            if tok.kind == "eof":
                return
            if tok.kind == "op":
                if tok.text == "(":
                    depth += 1
                elif tok.text == ")":
                    if depth == 0:
                        return
                    depth -= 1
                elif depth == 0 and tok.text == ";":
                    return
            self._advance()

    # -- 顶层结构 ----------------------------------------------------------

    def parse_single(self) -> Query:
        if self._peek().kind == "eof":
            raise ParseFatal(0, "SQL 为空。")
        query = self._parse_query(depth=1)
        if self._at_op(";"):
            self._advance()
        tok = self._peek()
        if tok.kind != "eof":
            raise ParseFatal(
                tok.pos,
                f"仅支持单条 SELECT 查询，检测到多余内容 {self._describe(tok)}。",
            )
        return query

    def _parse_query(self, depth: int) -> Query:
        pos = self._peek().pos
        ctes: list[Cte] = []
        if self._at_kw("WITH"):
            if depth > 1:
                self._problem(self._peek().pos, "不支持嵌套 WITH（仅支持一层）。")
            self._advance()
            if self._at_kw("RECURSIVE"):
                self._problem(self._peek().pos, "不支持递归 WITH（WITH RECURSIVE）。")
                self._advance()
            ctes = self._parse_cte_list(depth)
        select = self._parse_select(depth)
        while True:
            tok = self._peek()
            if tok.kind == "ident" and tok.upper in ("UNION", "EXCEPT", "INTERSECT"):
                self._problem(tok.pos, f"不支持复合查询 {tok.upper}。")
                self._advance()
                if self._at_kw("ALL") or self._at_kw("DISTINCT"):
                    self._advance()
                self._skip_to_statement_end()
                continue
            break
        return Query(pos, ctes, select)

    def _parse_cte_list(self, depth: int) -> list[Cte]:
        ctes: list[Cte] = []
        while True:
            name_tok = self._expect_ident("CTE 名称")
            columns: list[str] | None = None
            if self._at_op("("):
                self._advance()
                columns = []
                if not self._at_op(")"):
                    while True:
                        columns.append(self._expect_ident("CTE 列名").text)
                        if self._at_op(","):
                            self._advance()
                            continue
                        break
                self._expect_op(")")
            self._expect_kw("AS")
            self._expect_op("(")
            body = self._parse_query(depth + 1)
            self._expect_op(")")
            ctes.append(Cte(name_tok.pos, name_tok.text, columns, body))
            if self._at_op(","):
                self._advance()
                continue
            break
        return ctes

    def _parse_select(self, depth: int) -> Select:
        sel_tok = self._expect_kw("SELECT")
        if self._at_kw("DISTINCT"):
            self._problem(self._peek().pos, "不支持 DISTINCT。")
            self._advance()
        elif self._at_kw("ALL"):
            self._advance()
        items = [self._parse_item()]
        while self._at_op(","):
            self._advance()
            items.append(self._parse_item())
        sources: list[SourceRef] = []
        joins: list[Join] = []
        if self._at_kw("FROM"):
            self._advance()
            sources, joins = self._parse_from()
        where: Expr | None = None
        if self._at_kw("WHERE"):
            self._advance()
            where = self._parse_condition_guarded()
        self._parse_trailing_clauses()
        return Select(sel_tok.pos, items, sources, joins, where)

    def _parse_trailing_clauses(self) -> None:
        while True:
            tok = self._peek()
            if tok.kind == "ident" and tok.upper in (
                "GROUP", "HAVING", "ORDER", "LIMIT", "OFFSET", "WINDOW",
            ):
                self._problem(tok.pos, f"不支持 {tok.upper} 子句。")
                self._advance()
                if self._at_kw("BY"):
                    self._advance()
                self._skip_to_boundary()
                continue
            break

    def _parse_from(self) -> tuple[list[SourceRef], list[Join]]:
        sources = [self._parse_source_ref()]
        joins: list[Join] = []
        while True:
            if self._at_op(","):
                self._advance()
                sources.append(self._parse_source_ref())
                continue
            join_pos: int | None = None
            natural = False
            if self._at_kw("JOIN"):
                join_pos = self._peek().pos
                self._advance()
            elif self._at_kw("INNER") or self._at_kw("CROSS"):
                join_pos = self._peek().pos
                self._advance()
                self._expect_kw("JOIN")
            elif self._at_kw("LEFT"):
                join_pos = self._peek().pos
                self._advance()
                if self._at_kw("OUTER"):
                    self._advance()
                self._expect_kw("JOIN")
            elif self._at_kw("RIGHT") or self._at_kw("FULL"):
                tok = self._advance()
                join_pos = tok.pos
                self._problem(tok.pos, f"不支持 {tok.upper} JOIN。")
                if self._at_kw("OUTER"):
                    self._advance()
                self._expect_kw("JOIN")
            elif self._at_kw("NATURAL"):
                tok = self._advance()
                join_pos = tok.pos
                natural = True
                self._problem(
                    tok.pos, "不支持 NATURAL JOIN（隐式连接条件无法静态确认）。"
                )
                if self._at_kw("LEFT") or self._at_kw("INNER"):
                    self._advance()
                if self._at_kw("OUTER"):
                    self._advance()
                self._expect_kw("JOIN")
            else:
                break
            right = self._parse_source_ref()
            on: Expr | None = None
            if natural:
                on = UnknownExpr(join_pos)
            elif self._at_kw("ON"):
                self._advance()
                on = self._parse_condition_guarded()
            elif self._at_kw("USING"):
                tok = self._advance()
                self._problem(
                    tok.pos, "不支持 JOIN … USING（隐式连接条件无法静态确认）。"
                )
                if self._at_op("("):
                    self._advance()
                    self._skip_to_boundary()  # 停在 “)” 前
                    self._expect_op(")")
                on = UnknownExpr(join_pos)
            joins.append(Join(join_pos if join_pos is not None else right.pos, right, on))
            sources.append(right)
        return sources, joins

    def _parse_source_ref(self) -> SourceRef:
        tok = self._peek()
        if tok.kind == "op" and tok.text == "(":
            raise ParseFatal(
                tok.pos, "FROM 中不支持子查询或括号连接（仅支持表名或 CTE 名）。"
            )
        name = self._expect_ident("源表名称")
        if self._at_op("."):
            raise ParseFatal(self._peek().pos, "不支持 “库名.表名” 限定。")
        alias: str | None = None
        if self._at_kw("AS"):
            self._advance()
            alias = self._expect_ident("表别名", allow_string=True).text
        elif self._at_bare_alias():
            alias = self._advance().text
        return SourceRef(name.pos, name.text, alias)

    # -- 投影项 ------------------------------------------------------------

    def _parse_item(self) -> Item:
        pos = self._peek().pos
        try:
            if self._at_op("*"):
                self._advance()
                return StarItem(pos)
            if (
                self._peek().kind in ("ident", "qident")
                and self._peek(1).kind == "op"
                and self._peek(1).text == "."
                and self._peek(2).kind == "op"
                and self._peek(2).text == "*"
            ):
                table = self._advance().text
                self._advance()
                self._advance()
                return TableStarItem(pos, table)
            expr = self._parse_expr()
            alias: str | None = None
            if self._at_kw("AS"):
                self._advance()
                alias = self._expect_ident("列别名", allow_string=True).text
            elif self._at_bare_alias():
                alias = self._advance().text
            return ExprItem(pos, expr, alias)
        except _Unsupported as exc:
            self._problem(exc.pos, exc.message)
            self._skip_to_boundary(stop_comma=True)
            return UnknownItem(pos)

    # -- 条件（WHERE / JOIN ON） -------------------------------------------

    def _parse_condition_guarded(self) -> Expr:
        pos = self._peek().pos
        try:
            return self._parse_expr()
        except _Unsupported as exc:
            self._problem(exc.pos, exc.message)
            self._skip_to_boundary()
            return UnknownExpr(pos)

    # -- 表达式 ------------------------------------------------------------

    def _parse_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self._at_kw("OR"):
            tok = self._advance()
            right = self._parse_and()
            left = Logic(tok.pos, "OR", left, right)
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_comparison()
        while self._at_kw("AND"):
            tok = self._advance()
            right = self._parse_comparison()
            left = Logic(tok.pos, "AND", left, right)
        return left

    def _parse_comparison(self) -> Expr:
        left = self._parse_additive()
        tok = self._peek()
        if tok.kind == "op" and tok.text in _COMPARE_OPS:
            self._advance()
            right = self._parse_additive()
            return Compare(tok.pos, tok.text, left, right)
        if tok.kind == "ident":
            if tok.upper in (
                "IN", "LIKE", "GLOB", "BETWEEN", "IS", "MATCH", "REGEXP",
                "ISNULL", "NOTNULL", "COLLATE",
            ):
                raise _Unsupported(
                    tok.pos, f"不支持 {tok.upper} 条件（仅支持列比较与 AND/OR）。"
                )
            if tok.upper == "NOT" and self._peek(1).kind == "ident" and (
                self._peek(1).upper
                in ("IN", "LIKE", "GLOB", "BETWEEN", "MATCH", "REGEXP")
            ):
                raise _Unsupported(
                    tok.pos, f"不支持 NOT {self._peek(1).upper} 条件（仅支持列比较与 AND/OR）。"
                )
        return left

    def _parse_additive(self) -> Expr:
        left = self._parse_multiplicative()
        while True:
            tok = self._peek()
            if tok.kind == "op" and tok.text in ("+", "-"):
                self._advance()
                right = self._parse_multiplicative()
                left = Binary(tok.pos, tok.text, left, right)
            else:
                return left

    def _parse_multiplicative(self) -> Expr:
        left = self._parse_unary()
        while True:
            tok = self._peek()
            if tok.kind == "op" and tok.text in ("*", "/"):
                self._advance()
                right = self._parse_unary()
                left = Binary(tok.pos, tok.text, left, right)
            elif tok.kind == "op" and tok.text in ("%", "||", "&", "|", "<<", ">>"):
                raise _Unsupported(
                    tok.pos, f"不支持运算符 “{tok.text}”（仅支持四则运算 + - * /）。"
                )
            else:
                return left

    def _parse_unary(self) -> Expr:
        tok = self._peek()
        if tok.kind == "op" and tok.text in ("-", "+"):
            self._advance()
            return Unary(tok.pos, tok.text, self._parse_unary())
        if tok.kind == "ident" and tok.upper == "NOT":
            raise _Unsupported(tok.pos, "不支持 NOT（仅支持列比较与 AND/OR）。")
        return self._parse_atom()

    def _parse_atom(self) -> Expr:
        tok = self._peek()
        if tok.kind in ("number", "string"):
            self._advance()
            return Literal(tok.pos, tok.text)
        if tok.kind in ("ident", "qident"):
            if tok.kind == "ident":
                if tok.upper in ("NULL", "TRUE", "FALSE"):
                    self._advance()
                    return Literal(tok.pos, tok.text)
                if tok.upper == "CASE":
                    raise _Unsupported(tok.pos, "不支持 CASE 表达式。")
                if tok.upper in ("EXISTS", "CAST"):
                    raise _Unsupported(tok.pos, f"不支持 {tok.upper}。")
                if tok.upper in _NOT_REF_KEYWORDS:
                    raise ParseFatal(
                        tok.pos, f"表达式中出现意外的关键字 {tok.upper}。"
                    )
            self._advance()
            if self._at_op("("):
                raise _Unsupported(tok.pos, f"不支持函数调用 {tok.text}(…)。")
            if self._at_op("."):
                self._advance()
                col = self._expect_ident("列名")
                return ColumnRef(tok.pos, tok.text, col.text)
            return ColumnRef(tok.pos, None, tok.text)
        if tok.kind == "op" and tok.text == "(":
            self._advance()
            if self._at_kw("SELECT") or self._at_kw("WITH"):
                raise _Unsupported(tok.pos, "不支持子查询。")
            inner = self._parse_expr()
            self._expect_op(")")
            return inner
        raise ParseFatal(
            tok.pos, f"无法解析表达式（遇到 {self._describe(tok)}）。"
        )


def parse(text: str) -> tuple[Query, list[Problem]]:
    """兼容别名：解析单条 SELECT。"""
    return parse_query(text)
