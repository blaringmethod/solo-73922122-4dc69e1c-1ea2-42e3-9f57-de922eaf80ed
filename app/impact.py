"""字段影响静态核查：不下线前评估单条 SELECT 对源表列的依赖。

只静态解析，不执行 SQL：
* 逐列追溯每个结果列的来源（CTE 改名、表别名、同名列、通配符展开
  都按位置正确归属）；
* 报告依赖目标列的 JOIN ON / WHERE 条件（行集可能因此改变）；
* 任何无法消歧或超出支持范围的位置都会列入问题清单，
  此时整体结论绝不给出“无影响”。
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field

from . import select_sql
from .errors import ImpactError
from .select_sql import (
    Binary,
    ColumnRef,
    Compare,
    Cte,
    Expr,
    ExprItem,
    Literal,
    Logic,
    Query,
    Select,
    StarItem,
    TableStarItem,
    Unary,
    UnknownExpr,
    UnknownItem,
)
from .validation import validate_database

Dep = tuple[str, str]  # (基表, 列)


# ---------------------------------------------------------------------------
# 上传库的只读取用
# ---------------------------------------------------------------------------


def load_table_columns(db_path: str) -> dict[str, list[str]]:
    """读取各表列名（按定义顺序）。连接只能以只读方式打开。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out: dict[str, list[str]] = {}
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ):
            escaped = name.replace('"', '""')
            out[name] = [
                row[1] for row in conn.execute(f'PRAGMA table_info("{escaped}")')
            ]
        return out
    finally:
        conn.close()


@contextmanager
def validated_temp_db(raw: bytes):
    """把上传内容写入临时文件并校验，用完即删。

    分析全程只读上传库，不会改动上传原件或任务目录中的缩减结果。
    """
    fd, path = tempfile.mkstemp(prefix="impact_", suffix=".db")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        os.chmod(path, 0o444)
        schema = validate_database(path)
        columns = load_table_columns(path)
        yield schema, columns
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def normalize_targets(
    wanted: list[tuple[str, str]], columns: dict[str, list[str]]
) -> set[Dep]:
    """把用户选择的“表.列”规范为库中实际大小写，并校验存在性。"""
    tables_ci = {name.casefold(): name for name in columns}
    out: set[Dep] = set()
    for table, column in wanted:
        actual_t = tables_ci.get(table.casefold())
        if actual_t is None:
            raise ImpactError(f"目标表不存在：{table}。")
        cols_ci = {c.casefold(): c for c in columns[actual_t]}
        actual_c = cols_ci.get(column.casefold())
        if actual_c is None:
            raise ImpactError(f"表 {actual_t} 中不存在目标列：{column}。")
        out.add((actual_t, actual_c))
    return out


# ---------------------------------------------------------------------------
# 静态归因分析
# ---------------------------------------------------------------------------


def _line_col(text: str, pos: int) -> tuple[int, int]:
    line = text.count("\n", 0, pos) + 1
    last = text.rfind("\n", 0, pos)
    return line, pos - last


def _loc(text: str, pos: int) -> str:
    line, col = _line_col(text, pos)
    return f"第 {line} 行第 {col} 列"


@dataclass
class _Col:
    """某个来源暴露的一列；chains 记录到每个基表列的归因链。"""

    name: str
    chains: dict[Dep, tuple[str, ...]] = field(default_factory=dict)
    unknown: bool = False
    display: str = ""

    @property
    def deps(self) -> set[Dep]:
        return set(self.chains)


@dataclass
class _Source:
    exposed: str
    origin: str
    columns: list[_Col]


@dataclass
class _CteInfo:
    name: str
    columns: list[_Col]


@dataclass
class _Condition:
    kind: str  # JOIN ON / WHERE
    pos: int
    text: str
    chains: dict[Dep, tuple[str, ...]]
    unknown: bool


@dataclass
class _ResolvedSelect:
    output: list[_Col]
    conditions: list[_Condition]


def analyze_query(
    text: str, tables: dict[str, list[str]], targets: set[Dep]
) -> dict[str, object]:
    """静态分析单条 SELECT，返回可 JSON 序列化的报告。"""
    try:
        query, problems = select_sql.parse_query(text)
    except select_sql.ParseFatal as exc:
        raise ImpactError(f"{_loc(text, exc.pos)}：{exc.message}") from exc
    analyzer = _Analyzer(text, tables, targets, problems)
    return analyzer.run(query)


class _Analyzer:
    def __init__(
        self,
        text: str,
        tables: dict[str, list[str]],
        targets: set[Dep],
        problems: list[select_sql.Problem],
    ):
        self.text = text
        # 大小写不敏感地匹配表名/列名，展示时保留库中原始大小写
        self.tables = {k.casefold(): (k, v) for k, v in tables.items()}
        self.targets = set(targets)
        self.problems = list(problems)
        self._cte_asts: dict[str, Cte] = {}
        self._cte_infos: dict[str, _CteInfo] = {}
        self._resolving: set[str] = set()

    # -- 问题与错误 ---------------------------------------------------------

    def _problem(self, pos: int, message: str) -> None:
        self.problems.append(select_sql.Problem(pos, message))

    def _fatal(self, pos: int, message: str) -> None:
        raise ImpactError(f"{_loc(self.text, pos)}：{message}")

    # -- 入口 ---------------------------------------------------------------

    def run(self, query: Query) -> dict[str, object]:
        for cte in query.ctes:
            key = cte.name.casefold()
            if key in self._cte_asts:
                self._problem(cte.pos, f"CTE 名称“{cte.name}”重复定义。")
                continue
            self._cte_asts[key] = cte
        resolved = self._resolve_select(query.select, "主查询")
        return self._build_report(resolved)

    # -- CTE -----------------------------------------------------------------

    def _get_cte(self, key: str) -> _CteInfo | None:
        if key in self._cte_infos:
            return self._cte_infos[key]
        ast = self._cte_asts.get(key)
        if ast is None:
            return None
        if key in self._resolving:
            self._fatal(ast.pos, f"CTE“{ast.name}”存在递归引用，超出支持范围。")
        self._resolving.add(key)
        try:
            info = self._resolve_cte(ast)
            self._cte_infos[key] = info
            return info
        finally:
            self._resolving.discard(key)

    def _resolve_cte(self, cte: Cte) -> _CteInfo:
        # 嵌套 WITH 已在解析阶段记录问题，这里仍注册以便继续归因
        saved = dict(self._cte_asts)
        for nested in cte.body.ctes:
            self._cte_asts.setdefault(nested.name.casefold(), nested)
        try:
            resolved = self._resolve_select(cte.body.select, f"CTE {cte.name}")
        finally:
            self._cte_asts = saved
        renames = cte.columns or []
        if cte.columns is not None and len(renames) != len(resolved.output):
            self._problem(
                cte.pos,
                f"CTE“{cte.name}”的列名清单数量（{len(renames)}）"
                f"与输出列数（{len(resolved.output)}）不一致。",
            )
        columns: list[_Col] = []
        for idx, col in enumerate(resolved.output):
            if idx < len(renames):
                new_name = renames[idx]
                node = (
                    new_name
                    if new_name == col.name
                    else f"{new_name}（CTE 内原名 {col.name}）"
                )
            else:
                new_name = col.name
                node = col.name
            chains = {dep: (node,) + chain for dep, chain in col.chains.items()}
            columns.append(_Col(new_name, chains, col.unknown, col.display))
        return _CteInfo(cte.name, columns)

    # -- SELECT 解析 ---------------------------------------------------------

    def _resolve_select(self, sel: Select, context: str) -> _ResolvedSelect:
        sources = self._build_sources(sel.sources, context)
        conditions: list[_Condition] = []
        for join in sel.joins:
            if join.on is None:
                continue
            chains, unknown = self._resolve_expr(join.on, sources, context)
            conditions.append(
                _Condition("JOIN ON", join.on.pos, select_sql.render_expr(join.on),
                           chains, unknown)
            )
        if sel.where is not None:
            chains, unknown = self._resolve_expr(sel.where, sources, context)
            conditions.append(
                _Condition("WHERE", sel.where.pos, select_sql.render_expr(sel.where),
                           chains, unknown)
            )
        output: list[_Col] = []
        for item in sel.items:
            output.extend(self._expand_item(item, sources, context))
        return _ResolvedSelect(output, conditions)

    def _build_sources(self, refs, context: str) -> list[_Source]:
        sources: list[_Source] = []
        seen: set[str] = set()
        for ref in refs:
            key = ref.name.casefold()
            cte = self._get_cte(key)
            exposed = ref.alias or ref.name
            columns: list[_Col] = []
            if cte is not None:
                origin = f"CTE {cte.name}"
                for c in cte.columns:
                    node = f"{exposed}.{c.name}"
                    chains = {
                        dep: (node,) + chain for dep, chain in c.chains.items()
                    }
                    columns.append(_Col(c.name, chains, c.unknown, node))
            else:
                base = self.tables.get(key)
                if base is None:
                    self._fatal(
                        ref.pos, f"{context}：FROM 中找不到表或 CTE“{ref.name}”。"
                    )
                actual, colnames = base
                origin = f"表 {actual}"
                for cn in colnames:
                    node = f"{exposed}.{cn}"
                    columns.append(
                        _Col(cn, {(actual, cn): (node, f"基表列 {actual}.{cn}")},
                             display=node)
                    )
            ekey = exposed.casefold()
            if ekey in seen:
                self._problem(
                    ref.pos,
                    f"{context}：来源名称“{exposed}”重复出现，列归属可能不准确。",
                )
            seen.add(ekey)
            sources.append(_Source(exposed, origin, columns))
        if len(sources) > 2:
            self._problem(
                refs[2].pos,
                f"{context}：源表数量为 {len(refs)} 张，超出支持范围（至多两张）。",
            )
        return sources

    # -- 表达式归因 -----------------------------------------------------------

    def _resolve_expr(
        self, expr: Expr, sources: list[_Source], context: str
    ) -> tuple[dict[Dep, tuple[str, ...]], bool]:
        if isinstance(expr, ColumnRef):
            return self._resolve_column(expr, sources, context)
        if isinstance(expr, Literal):
            return {}, False
        if isinstance(expr, Unary):
            return self._resolve_expr(expr.operand, sources, context)
        if isinstance(expr, (Binary, Compare, Logic)):
            left_chains, left_unknown = self._resolve_expr(expr.left, sources, context)
            right_chains, right_unknown = self._resolve_expr(expr.right, sources, context)
            merged = dict(left_chains)
            for dep, chain in right_chains.items():
                merged.setdefault(dep, chain)
            return merged, left_unknown or right_unknown
        if isinstance(expr, UnknownExpr):
            return {}, True
        return {}, True  # pragma: no cover - 防御性兜底

    def _resolve_column(
        self, ref: ColumnRef, sources: list[_Source], context: str
    ) -> tuple[dict[Dep, tuple[str, ...]], bool]:
        if ref.table is not None:
            for src in sources:
                if src.exposed.casefold() == ref.table.casefold():
                    for col in src.columns:
                        if col.name.casefold() == ref.name.casefold():
                            return dict(col.chains), col.unknown
                    self._problem(
                        ref.pos,
                        f"{context}：{src.exposed} 中没有列“{ref.name}”。",
                    )
                    return {}, True
            self._problem(
                ref.pos, f"{context}：找不到表或别名“{ref.table}”。"
            )
            return {}, True
        matches: list[tuple[_Source, _Col]] = [
            (src, col)
            for src in sources
            for col in src.columns
            if col.name.casefold() == ref.name.casefold()
        ]
        if not matches:
            self._problem(ref.pos, f"{context}：找不到列“{ref.name}”。")
            return {}, True
        if len(matches) > 1:
            owners = "、".join(
                f"{src.exposed}.{col.name}" for src, col in matches
            )
            self._problem(
                ref.pos,
                f"{context}：列“{ref.name}”同时存在于 {owners}，无法消歧。",
            )
            return {}, True
        col = matches[0][1]
        return dict(col.chains), col.unknown

    # -- 投影项展开 -----------------------------------------------------------

    def _expand_item(
        self, item, sources: list[_Source], context: str
    ) -> list[_Col]:
        if isinstance(item, StarItem):
            if not sources:
                self._problem(item.pos, f"{context}：SELECT * 没有 FROM 来源，无法展开。")
                return [_Col("（无法展开）", unknown=True, display="*")]
            out: list[_Col] = []
            for src in sources:
                for col in src.columns:
                    out.append(_Col(col.name, dict(col.chains), col.unknown,
                                    col.display))
            return out
        if isinstance(item, TableStarItem):
            for src in sources:
                if src.exposed.casefold() == item.table.casefold():
                    return [
                        _Col(col.name, dict(col.chains), col.unknown, col.display)
                        for col in src.columns
                    ]
            self._problem(
                item.pos,
                f"{context}：找不到表或别名“{item.table}”，无法展开 {item.table}.*。",
            )
            return [_Col("（无法展开）", unknown=True, display=f"{item.table}.*")]
        if isinstance(item, ExprItem):
            chains, unknown = self._resolve_expr(item.expr, sources, context)
            display = select_sql.render_expr(item.expr)
            if item.alias:
                name = item.alias
            elif isinstance(item.expr, ColumnRef):
                name = item.expr.name
            else:
                name = display
            if not isinstance(item.expr, ColumnRef):
                chains = {
                    dep: (display,) + chain for dep, chain in chains.items()
                }
            return [_Col(name, chains, unknown, display)]
        if isinstance(item, UnknownItem):
            return [_Col("（无法解析）", unknown=True, display="（无法解析）")]
        return [_Col("（无法解析）", unknown=True, display="（无法解析）")]  # pragma: no cover

    # -- 报告 -----------------------------------------------------------------

    def _build_report(self, resolved: _ResolvedSelect) -> dict[str, object]:
        columns: list[dict[str, object]] = []
        for idx, col in enumerate(resolved.output, 1):
            deps = sorted(col.deps)
            matched = sorted(set(deps) & self.targets)
            status = (
                "unknown"
                if col.unknown
                else ("affected" if matched else "unaffected")
            )
            trace = [
                " ← ".join((f'结果列 {idx}（{col.name}）',) + chain)
                for chain in col.chains.values()
            ]
            columns.append(
                {
                    "pos": idx,
                    "name": col.name,
                    "expression": col.display,
                    "status": status,
                    "sources": [
                        {"table": t, "column": c} for t, c in deps
                    ],
                    "matched_targets": [
                        {"table": t, "column": c} for t, c in matched
                    ],
                    "trace": trace,
                }
            )

        conditions: list[dict[str, object]] = []
        for cond in resolved.conditions:
            deps = sorted(cond.chains)
            matched = sorted(set(deps) & self.targets)
            status = (
                "unknown"
                if cond.unknown
                else ("affected" if matched else "unaffected")
            )
            conditions.append(
                {
                    "kind": cond.kind,
                    "location": _loc(self.text, cond.pos),
                    "text": cond.text,
                    "status": status,
                    "sources": [
                        {"table": t, "column": c} for t, c in deps
                    ],
                    "matched_targets": [
                        {"table": t, "column": c} for t, c in matched
                    ],
                }
            )

        problems = sorted({(p.pos, p.message) for p in self.problems})
        problem_list = [
            {"location": _loc(self.text, pos), "message": message}
            for pos, message in problems
        ]

        affected_columns = [c for c in columns if c["status"] == "affected"]
        affected_conditions = [c for c in conditions if c["status"] == "affected"]
        has_unknown = (
            any(c["status"] == "unknown" for c in columns)
            or any(c["status"] == "unknown" for c in conditions)
            or bool(problem_list)
        )
        if affected_columns or affected_conditions:
            verdict = "affected"
            verdict_text = (
                f"检测到依赖：{len(affected_columns)} 个结果列的值、"
                f"{len(affected_conditions)} 个 JOIN/WHERE 条件（行集）"
                "可能受目标列影响。"
            )
        elif has_unknown:
            verdict = "unknown"
            verdict_text = (
                "存在无法静态确认的位置（见问题清单），"
                "不能排除对目标列的影响，不给出“无影响”结论。"
            )
        else:
            verdict = "no_impact"
            verdict_text = (
                "在所支持的语法范围内完成静态追溯，"
                "未发现结果列或 JOIN/WHERE 条件依赖目标列。"
            )

        return {
            "verdict": verdict,
            "verdict_text": verdict_text,
            "targets": [
                {"table": t, "column": c} for t, c in sorted(self.targets)
            ],
            "columns": columns,
            "conditions": conditions,
            "problems": problem_list,
            "stats": {
                "column_count": len(columns),
                "affected_columns": len(affected_columns),
                "condition_count": len(conditions),
                "affected_conditions": len(affected_conditions),
                "problem_count": len(problem_list),
            },
        }
