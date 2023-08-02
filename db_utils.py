"""Utils for database."""
import os
import psycopg
from db_config import DBInfo


def get_and_reserve_params():
    # Create a DB
    db_name = DBInfo().db_name
    with psycopg.connect(dbname=db_name, password="postgres", autocommit=True) as con, con.cursor(row_factory=psycopg.rows.dict_row) as cur:
        res = cur.execute("UPDATE metadata SET reserved=True WHERE id in (SELECT id from metadata WHERE reserved=False LIMIT 1) RETURNING *")
        data = res.fetchone()
    return data


def record_performance(results):
    """Update results and set status of the given metadata row."""
    db_name = DBInfo().db_name
    with psycopg.connect(dbname=db_name, password="postgres", autocommit=True) as con, con.cursor(row_factory=psycopg.rows.dict_row) as cur:
        # First update results
        cols = [x for x in results.keys()]
        vals = [x for x in results.values()]
        query = """INSERT INTO results(%s) VALUES (%%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s); """ % (', '.join(cols))
        cur.execute(query, vals)

        # Next update the metadata
        query = """UPDATE metadata SET reserved=True, finished=True where id=%s"""
        cur.execute(query, (results["meta_id"],))
    print("Updated results.")


def get_all_results():
    """Select results joined with their metadata."""
    db_name = DBInfo().db_name
    with psycopg.connect(dbname=db_name, password="postgres", autocommit=True) as con, con.cursor(row_factory=psycopg.rows.dict_row) as cur:
        query = """SELECT * FROM metadata INNER JOIN results ON results.meta_id = metadata.id;"""
        res = cur.execute(query)
        data = res.fetchall()
    return data

