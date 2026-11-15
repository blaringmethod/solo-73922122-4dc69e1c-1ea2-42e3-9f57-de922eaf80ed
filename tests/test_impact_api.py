"""字段影响核查 API 测试。"""
from __future__ import annotations

import glob
import os
import sqlite3
import tempfile

# 与 tests/test_api.py 相同的夹具模式：测试进程内使用临时数据目录
os.environ.setdefault("DATA_DIR", "/tmp/reducer-test-data")

import pytest  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def shop_db(tmp_path):
    """两张表、含外键的合规库。"""
    path = tmp_path / "shop.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE dept (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE employee (
            id INTEGER PRIMARY KEY,
            dept_id INTEGER NOT NULL REFERENCES dept(id),
            mentor_id INTEGER
        );
        INSERT INTO dept VALUES (1, 'Platform'), (2, 'QA');
        INSERT INTO employee VALUES (10, 1, NULL), (11, 2, 10);
        """
    )
    conn.commit()
    conn.close()
    return path


QUERY = """\
WITH team AS (
  SELECT e.id AS emp_id, e.dept_id AS dep
  FROM employee e
)
SELECT t.emp_id, t.dep, d.name
FROM team t JOIN dept d ON t.dep = d.id
WHERE d.name = 'Platform' OR t.dep > 1;
"""


def post_impact(client, db_bytes, query, targets):
    return client.post(
        "/api/impact",
        files={"db": ("x.db", db_bytes, "application/x-sqlite3")},
        data={"query": query, "targets": targets},
    )


def test_schema_endpoint(client, shop_db):
    r = client.post("/api/impact/schema", files={"db": ("x.db", shop_db.read_bytes())})
    assert r.status_code == 200
    body = r.json()
    assert body["tables"][0]["name"] == "dept"
    assert body["tables"][1]["columns"] == ["id", "dept_id", "mentor_id"]


def test_schema_rejects_bad_file(client):
    r = client.post(
        "/api/impact/schema",
        files={"db": ("x.db", b"hello", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_impact_full_flow(client, shop_db):
    r = post_impact(client, shop_db.read_bytes(), QUERY, ["employee.dept_id"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "affected"
    cols = body["columns"]
    assert [c["pos"] for c in cols] == [1, 2, 3]
    assert cols[1]["name"] == "dep"
    assert cols[1]["status"] == "affected"
    assert cols[1]["sources"] == [{"table": "employee", "column": "dept_id"}]
    kinds = {c["kind"] for c in body["conditions"]}
    assert kinds == {"JOIN ON", "WHERE"}
    assert all(c["status"] == "affected" for c in body["conditions"])


def test_impact_no_impact(client, shop_db):
    r = post_impact(client, shop_db.read_bytes(),
                    "SELECT e.id FROM employee e", ["employee.dept_id"])
    assert r.status_code == 200
    assert r.json()["verdict"] == "no_impact"


def test_impact_unsupported_never_no_impact(client, shop_db):
    r = post_impact(client, shop_db.read_bytes(),
                    "SELECT mentor_id FROM employee GROUP BY mentor_id",
                    ["employee.dept_id"])
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "unknown"
    assert body["problems"]
    for p in body["problems"]:
        assert "第" in p["location"] and "行" in p["location"]


def test_impact_rejects_non_select(client, shop_db):
    r = post_impact(client, shop_db.read_bytes(),
                    "DELETE FROM employee;", ["employee.dept_id"])
    assert r.status_code == 400
    assert "SELECT" in r.json()["detail"]


def test_impact_rejects_multiple_statements(client, shop_db):
    r = post_impact(client, shop_db.read_bytes(),
                    "SELECT id FROM employee; SELECT id FROM employee;",
                    ["employee.dept_id"])
    assert r.status_code == 400


def test_impact_rejects_missing_target(client, shop_db):
    r = post_impact(client, shop_db.read_bytes(),
                    "SELECT id FROM employee", ["employee.nosuch"])
    assert r.status_code == 400
    assert "nosuch" in r.json()["detail"]


def test_impact_rejects_empty_query(client, shop_db):
    r = post_impact(client, shop_db.read_bytes(), "   ", ["employee.dept_id"])
    assert r.status_code == 400


def test_impact_rejects_empty_targets(client, shop_db):
    r = client.post("/api/impact", files={"db": ("x.db", shop_db.read_bytes())},
                    data={"query": "SELECT 1"})
    assert r.status_code == 400


def test_impact_does_not_modify_upload_or_leave_temp(client, shop_db):
    before = shop_db.read_bytes()
    r = post_impact(client, before, QUERY, ["employee.dept_id"])
    assert r.status_code == 200
    # 上传库本身不会被修改
    assert shop_db.read_bytes() == before
    # 不留下用于分析的临时文件
    leftovers = glob.glob(os.path.join(tempfile.gettempdir(), "impact_*"))
    assert leftovers == []


def test_impact_rejects_bad_db(client):
    r = client.post(
        "/api/impact",
        files={"db": ("x.db", b"hello", "application/octet-stream")},
        data={"query": "SELECT 1", "targets": ["a.b"]},
    )
    assert r.status_code == 400


def test_query_too_large(client, shop_db):
    r = post_impact(client, shop_db.read_bytes(),
                    "SELECT " + "1," * 200000 + "1 FROM employee",
                    ["employee.dept_id"])
    assert r.status_code == 400
