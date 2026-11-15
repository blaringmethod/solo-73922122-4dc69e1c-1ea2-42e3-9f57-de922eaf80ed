"""字段影响核查：解析器与静态归因分析测试。"""
from __future__ import annotations

import pytest

from app.errors import ImpactError
from app.impact import analyze_query

TABLES = {
    "dept": ["id", "name"],
    "employee": ["id", "dept_id", "mentor_id"],
    "project": ["id", "owner_id", "dept_id"],
}
T_DEPT_ID = {("employee", "dept_id")}


def analyze(sql, targets=T_DEPT_ID, tables=TABLES):
    return analyze_query(sql, tables, targets)


def col(report, pos):
    return report["columns"][pos - 1]


def cond(report, kind):
    return [c for c in report["conditions"] if c["kind"] == kind]


# ---------------------------------------------------------------------------
# 基本投影归因
# ---------------------------------------------------------------------------


def test_direct_column_affected():
    r = analyze("SELECT e.dept_id FROM employee e")
    assert r["verdict"] == "affected"
    assert col(r, 1)["status"] == "affected"
    assert col(r, 1)["sources"] == [{"table": "employee", "column": "dept_id"}]


def test_other_column_unaffected():
    r = analyze("SELECT e.id FROM employee e")
    assert r["verdict"] == "no_impact"
    assert col(r, 1)["status"] == "unaffected"


def test_alias_keeps_attribution():
    r = analyze("SELECT e.dept_id AS d FROM employee e")
    assert col(r, 1)["name"] == "d"
    assert col(r, 1)["status"] == "affected"


def test_arithmetic_projection():
    r = analyze("SELECT e.dept_id * 2 + 1 AS x FROM employee e")
    assert col(r, 1)["status"] == "affected"
    assert col(r, 1)["sources"] == [{"table": "employee", "column": "dept_id"}]


def test_arithmetic_multi_source():
    r = analyze("SELECT e.id + e.mentor_id AS y FROM employee e")
    assert col(r, 1)["status"] == "unaffected"
    assert {s["column"] for s in col(r, 1)["sources"]} == {"id", "mentor_id"}
    assert r["verdict"] == "no_impact"


def test_literal_projection_no_deps():
    r = analyze("SELECT 1 AS one, e.dept_id FROM employee e")
    assert col(r, 1)["status"] == "unaffected"
    assert col(r, 1)["sources"] == []
    assert col(r, 2)["status"] == "affected"


def test_unqualified_column_resolves():
    r = analyze("SELECT dept_id FROM employee")
    assert col(r, 1)["status"] == "affected"


def test_case_insensitive_names():
    r = analyze("SELECT E.DEPT_ID FROM EMPLOYEE E")
    assert col(r, 1)["status"] == "affected"
    assert col(r, 1)["sources"] == [{"table": "employee", "column": "dept_id"}]


def test_quoted_identifiers():
    r = analyze('SELECT "dept_id" FROM "employee"')
    assert col(r, 1)["status"] == "affected"


# ---------------------------------------------------------------------------
# 同名列与消歧
# ---------------------------------------------------------------------------


def test_same_name_columns_qualified():
    sql = "SELECT a.id, b.id FROM project a JOIN employee b ON a.owner_id = b.id"
    r = analyze(sql, targets={("employee", "id")})
    assert col(r, 1)["status"] == "unaffected"
    assert col(r, 1)["sources"] == [{"table": "project", "column": "id"}]
    assert col(r, 2)["status"] == "affected"
    assert col(r, 2)["sources"] == [{"table": "employee", "column": "id"}]


def test_same_name_columns_unqualified_ambiguous():
    sql = "SELECT id FROM project a JOIN employee b ON a.owner_id = b.id"
    r = analyze(sql, targets={("employee", "id")})
    assert col(r, 1)["status"] == "unknown"
    assert r["verdict"] != "no_impact"  # 绝不给出“无影响”
    assert any("无法消歧" in p["message"] for p in r["problems"])
    assert any("id" in p["message"] for p in r["problems"])


def test_ambiguity_alone_forces_unknown_verdict():
    # 没有其它依赖时，仅歧义就足以让结论变为“无法确定”
    r = analyze("SELECT id FROM employee a, project b", targets={("employee", "id")})
    assert col(r, 1)["status"] == "unknown"
    assert r["verdict"] == "unknown"


def test_ambiguity_reports_position():
    sql = "SELECT id FROM project a JOIN employee b ON a.owner_id = b.id"
    r = analyze(sql)
    problem = r["problems"][0]
    assert "第 1 行第 8 列" == problem["location"]


# ---------------------------------------------------------------------------
# 通配符展开
# ---------------------------------------------------------------------------


def test_star_expansion_positions():
    r = analyze("SELECT * FROM employee")
    assert [c["name"] for c in r["columns"]] == ["id", "dept_id", "mentor_id"]
    assert col(r, 1)["status"] == "unaffected"
    assert col(r, 2)["status"] == "affected"
    assert col(r, 3)["status"] == "unaffected"


def test_table_star_expansion():
    r = analyze("SELECT e.* FROM employee e")
    assert len(r["columns"]) == 3
    assert col(r, 2)["status"] == "affected"


def test_star_join_duplicate_names_attributed():
    sql = "SELECT * FROM project a JOIN employee b ON a.owner_id = b.id"
    r = analyze(sql, targets={("employee", "id")})
    # project(id, owner_id, dept_id) + employee(id, dept_id, mentor_id)
    assert [c["name"] for c in r["columns"]] == [
        "id", "owner_id", "dept_id", "id", "dept_id", "mentor_id",
    ]
    assert col(r, 1)["sources"] == [{"table": "project", "column": "id"}]
    assert col(r, 4)["sources"] == [{"table": "employee", "column": "id"}]
    assert col(r, 4)["status"] == "affected"
    assert col(r, 1)["status"] == "unaffected"


# ---------------------------------------------------------------------------
# CTE
# ---------------------------------------------------------------------------


def test_cte_rename_attribution():
    sql = ("WITH c AS (SELECT e.dept_id AS d FROM employee e) "
           "SELECT c.d AS x FROM c")
    r = analyze(sql)
    assert col(r, 1)["name"] == "x"
    assert col(r, 1)["status"] == "affected"
    assert col(r, 1)["sources"] == [{"table": "employee", "column": "dept_id"}]
    trace = col(r, 1)["trace"][0]
    assert "c.d" in trace and "d" in trace and "employee.dept_id" in trace


def test_cte_column_list_rename():
    sql = "WITH c(x, y) AS (SELECT id, dept_id FROM employee) SELECT x, y FROM c"
    r = analyze(sql)
    assert col(r, 1)["name"] == "x"
    assert col(r, 1)["status"] == "unaffected"
    assert col(r, 2)["name"] == "y"
    assert col(r, 2)["status"] == "affected"
    assert col(r, 2)["sources"] == [{"table": "employee", "column": "dept_id"}]


def test_cte_chained_reference():
    sql = ("WITH a AS (SELECT dept_id FROM employee), "
           "b AS (SELECT dept_id FROM a) SELECT dept_id FROM b")
    r = analyze(sql)
    assert col(r, 1)["status"] == "affected"
    assert col(r, 1)["sources"] == [{"table": "employee", "column": "dept_id"}]


def test_cte_star_inside_and_outside():
    r = analyze("WITH c AS (SELECT * FROM employee) SELECT * FROM c")
    assert [c["name"] for c in r["columns"]] == ["id", "dept_id", "mentor_id"]
    assert col(r, 2)["status"] == "affected"


def test_cte_shadows_base_table():
    # CTE 名与基表同名时，外部引用的是 CTE；其 dept_id 实际来自 project.owner_id
    sql = ("WITH employee AS (SELECT owner_id AS dept_id FROM project) "
           "SELECT dept_id FROM employee")
    r = analyze(sql)
    assert col(r, 1)["sources"] == [{"table": "project", "column": "owner_id"}]
    assert r["verdict"] == "no_impact"


def test_cte_body_self_name_is_circular():
    # 与 SQLite 行为一致：CTE 体内引用自身名称算循环引用
    with pytest.raises(ImpactError, match="递归"):
        analyze("WITH employee AS (SELECT dept_id FROM employee) "
                "SELECT dept_id FROM employee")


def test_recursive_cte_flagged():
    r = analyze("WITH RECURSIVE c AS (SELECT dept_id FROM employee) SELECT * FROM c")
    assert any("递归" in p["message"] for p in r["problems"])
    assert r["verdict"] != "no_impact"


def test_self_referencing_cte_rejected():
    with pytest.raises(ImpactError, match="递归"):
        analyze("WITH c AS (SELECT * FROM c) SELECT * FROM c")


def test_nested_with_flagged():
    sql = ("WITH c AS (WITH d AS (SELECT dept_id FROM employee) "
           "SELECT * FROM d) SELECT * FROM c")
    r = analyze(sql)
    assert any("嵌套" in p["message"] for p in r["problems"])
    assert r["verdict"] != "no_impact"


# ---------------------------------------------------------------------------
# JOIN / WHERE 条件
# ---------------------------------------------------------------------------


def test_join_on_dependency():
    sql = "SELECT d.name FROM dept d JOIN employee e ON e.dept_id = d.id"
    r = analyze(sql)
    assert col(r, 1)["status"] == "unaffected"
    join_conds = cond(r, "JOIN ON")
    assert len(join_conds) == 1
    assert join_conds[0]["status"] == "affected"
    assert join_conds[0]["matched_targets"] == [
        {"table": "employee", "column": "dept_id"}
    ]
    assert r["verdict"] == "affected"  # 行集可能改变


def test_join_on_unaffected():
    sql = "SELECT d.name FROM dept d JOIN employee e ON e.mentor_id = d.id"
    r = analyze(sql)
    assert cond(r, "JOIN ON")[0]["status"] == "unaffected"
    assert r["verdict"] == "no_impact"


def test_where_and_or():
    sql = ("SELECT e.id FROM employee e "
           "WHERE e.dept_id = 2 AND e.id > 10 OR e.mentor_id = 3")
    r = analyze(sql)
    where = cond(r, "WHERE")
    assert len(where) == 1
    assert where[0]["status"] == "affected"


def test_where_unaffected():
    r = analyze("SELECT e.id FROM employee e WHERE e.id = 2")
    assert cond(r, "WHERE")[0]["status"] == "unaffected"
    assert r["verdict"] == "no_impact"


def test_where_string_compare():
    r = analyze("SELECT d.id FROM dept d WHERE d.name = 'Platform'")
    assert cond(r, "WHERE")[0]["status"] == "unaffected"
    assert r["verdict"] == "no_impact"


def test_comma_join_where_dependency():
    sql = "SELECT a.id FROM employee a, dept b WHERE a.dept_id = b.id"
    r = analyze(sql)
    assert cond(r, "WHERE")[0]["status"] == "affected"
    assert r["verdict"] == "affected"


def test_condition_location_reported():
    sql = "SELECT e.id FROM employee e WHERE e.dept_id = 2"
    r = analyze(sql)
    assert cond(r, "WHERE")[0]["location"].startswith("第 1 行")


# ---------------------------------------------------------------------------
# 超出支持范围：指出位置，绝不给出“无影响”
# ---------------------------------------------------------------------------


def test_function_call_flagged():
    r = analyze("SELECT COUNT(id) FROM employee")
    assert col(r, 1)["status"] == "unknown"
    assert any("函数" in p["message"] for p in r["problems"])
    assert r["verdict"] == "unknown"


def test_group_by_flagged():
    r = analyze("SELECT mentor_id FROM employee GROUP BY mentor_id")
    assert any("GROUP" in p["message"] for p in r["problems"])
    assert r["verdict"] == "unknown"  # 投影虽不依赖目标列，仍不得“无影响”


def test_distinct_flagged():
    r = analyze("SELECT mentor_id FROM employee WHERE 1 = 1", targets=T_DEPT_ID)
    assert r["verdict"] == "no_impact"  # 对照：完全在支持范围内
    r2 = analyze("SELECT DISTINCT mentor_id FROM employee")
    assert r2["verdict"] == "unknown"


def test_not_condition_flagged():
    r = analyze("SELECT id FROM employee WHERE NOT dept_id = 1")
    assert cond(r, "WHERE")[0]["status"] == "unknown"
    assert r["verdict"] == "unknown"


def test_in_condition_flagged():
    r = analyze("SELECT id FROM employee WHERE dept_id IN (1, 2)")
    assert any("IN" in p["message"] for p in r["problems"])
    assert r["verdict"] == "unknown"


def test_subquery_from_rejected():
    with pytest.raises(ImpactError) as exc:
        analyze("SELECT x FROM (SELECT dept_id FROM employee) AS x")
    assert "第 1 行" in str(exc.value)


def test_exists_flagged():
    r = analyze("SELECT id FROM employee WHERE EXISTS (SELECT 1 FROM dept)")
    assert r["verdict"] == "unknown"


def test_case_expr_flagged():
    r = analyze("SELECT CASE WHEN id > 1 THEN dept_id END FROM employee")
    assert col(r, 1)["status"] == "unknown"
    assert r["verdict"] == "unknown"


def test_union_flagged():
    r = analyze("SELECT dept_id FROM employee UNION SELECT id FROM employee")
    assert any("UNION" in p["message"] for p in r["problems"])
    assert r["verdict"] != "no_impact"


def test_three_tables_flagged_but_attributed():
    sql = ("SELECT e.dept_id FROM employee e "
           "JOIN dept d ON e.dept_id = d.id "
           "JOIN project p ON p.owner_id = e.id")
    r = analyze(sql)
    assert any("两张" in p["message"] for p in r["problems"])
    assert col(r, 1)["status"] == "affected"


def test_modulo_operator_flagged():
    r = analyze("SELECT dept_id % 2 FROM employee")
    assert any("四则运算" in p["message"] for p in r["problems"])
    assert r["verdict"] == "unknown"


def test_natural_join_flagged():
    r = analyze("SELECT a.id FROM employee a NATURAL JOIN dept b")
    assert any("NATURAL" in p["message"] for p in r["problems"])
    assert r["verdict"] == "unknown"


def test_join_using_flagged():
    r = analyze("SELECT a.id FROM employee a JOIN dept b USING (id)")
    assert any("USING" in p["message"] for p in r["problems"])
    assert r["verdict"] == "unknown"


def test_order_by_flagged():
    r = analyze("SELECT mentor_id FROM employee ORDER BY mentor_id")
    assert any("ORDER" in p["message"] for p in r["problems"])
    assert r["verdict"] == "unknown"


# ---------------------------------------------------------------------------
# 结构性错误（400）
# ---------------------------------------------------------------------------


def test_multiple_statements_rejected():
    with pytest.raises(ImpactError, match="单条"):
        analyze("SELECT dept_id FROM employee; SELECT id FROM employee;")


def test_non_select_rejected():
    with pytest.raises(ImpactError):
        analyze("DELETE FROM employee;")


def test_unknown_table_rejected():
    with pytest.raises(ImpactError, match="missing_table"):
        analyze("SELECT x FROM missing_table")


def test_unknown_column_flagged():
    r = analyze("SELECT nosuch FROM employee")
    assert col(r, 1)["status"] == "unknown"
    assert r["verdict"] == "unknown"


def test_empty_sql_rejected():
    with pytest.raises(ImpactError):
        analyze("   ")


# ---------------------------------------------------------------------------
# 多目标列
# ---------------------------------------------------------------------------


def test_multiple_targets():
    r = analyze(
        "SELECT id, dept_id, mentor_id FROM employee",
        targets={("employee", "dept_id"), ("employee", "mentor_id")},
    )
    assert col(r, 1)["status"] == "unaffected"
    assert col(r, 2)["status"] == "affected"
    assert col(r, 3)["status"] == "affected"
    assert r["verdict"] == "affected"


def test_target_normalization_in_report():
    r = analyze("SELECT id FROM employee")
    assert r["targets"] == [{"table": "employee", "column": "dept_id"}]
