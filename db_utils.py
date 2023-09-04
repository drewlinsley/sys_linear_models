"""Utils for database."""
import os
import sys
import json
import psycopg
from db_config import DBInfo, CPInfo


def get_and_reserve_params(cp=False):
    # Create a DB
    if cp:
        db_name = CPInfo().db_name
    else:
        db_name = DBInfo().db_name
    with psycopg.connect(dbname=db_name, password="postgres", autocommit=True) as con, con.cursor(row_factory=psycopg.rows.dict_row) as cur:
        # res = cur.execute("UPDATE metadata SET reserved=True WHERE id in (SELECT id from metadata WHERE reserved=False ORDER BY random() LIMIT 1) RETURNING *")
        res = cur.execute("UPDATE metadata SET reserved=True WHERE id in (SELECT id from metadata WHERE reserved=False LIMIT 1) RETURNING *")
        data = res.fetchone()
    return data


def record_performance(results, cp=False):
    """Update results and set status of the given metadata row."""
    if cp:
        db_name = CPInfo().db_name
    else:
        db_name = DBInfo().db_name
    with psycopg.connect(dbname=db_name, password="postgres", autocommit=True) as con, con.cursor(row_factory=psycopg.rows.dict_row) as cur:
        # First update results
        cols = [x for x in results.keys()]
        vals = [x for x in results.values()]
        query = """INSERT INTO results(%s) VALUES (%%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s); """ % (', '.join(cols))
        cur.execute(query, vals)

        # Next update the metadata
        query = """UPDATE metadata SET reserved=True, finished=True where id=%s"""
        cur.execute(query, (results["meta_id"],))
    print("Updated results.")


def get_all_results(cp=False):
    """Select results joined with their metadata."""
    if cp:
        db_name = CPInfo().db_name
    else:
        db_name = DBInfo().db_name
    with psycopg.connect(dbname=db_name, password="postgres", autocommit=True) as con, con.cursor(row_factory=psycopg.rows.dict_row) as cur:
        query = """SELECT * FROM metadata INNER JOIN results ON results.meta_id = metadata.id;"""
        res = cur.execute(query)
        data = res.fetchall()
    return data


def get_all_meta(cp=False):
    """Select results joined with their metadata."""
    if cp:
        db_name = CPInfo().db_name
    else:
        db_name = DBInfo().db_name
    with psycopg.connect(dbname=db_name, password="postgres", autocommit=True) as con, con.cursor(row_factory=psycopg.rows.dict_row) as cur:
        query = """SELECT * FROM metadata ORDER BY id;"""
        # query = """SELECT * FROM metadata WHERE reserved=True and finished=False ORDER BY id ;"""
        res = cur.execute(query)
        data = res.fetchall()
    return data


def find_unfinished(cp=False):
    """Select results joined with their metadata."""
    if cp:
        db_name = CPInfo().db_name
    else:
        db_name = DBInfo().db_name
    with psycopg.connect(dbname=db_name, password="postgres", autocommit=True) as con, con.cursor(row_factory=psycopg.rows.dict_row) as cur:
        query = """SELECT * FROM metadata WHERE reserved=True and finished=False ORDER BY id ;"""
        res = cur.execute(query)
        data = res.fetchall()
    return data


def reset_unfinished():
    """Set the reserved flag=False for entries in metadata where reserved=True and finished=False."""
    db_name = DBInfo().db_name
    with psycopg.connect(dbname=db_name, password="postgres", autocommit=True) as con, con.cursor(row_factory=psycopg.rows.dict_row) as cur:
        query = """UPDATE metadata SET reserved=False where reserved=True and finished=False;"""
        res = cur.execute(query)


def clean_db():
    """Delete rows where finished != True."""
    db_name = DBInfo().db_name
    with psycopg.connect(dbname=db_name, password="postgres", autocommit=True) as con, con.cursor(row_factory=psycopg.rows.dict_row) as cur:
        query = """DELETE FROM metadata WHERE finished=False;"""
        res = cur.execute(query)


def dump_db(name):
    db_name = DBInfo().db_name
    with psycopg.connect(dbname=db_name, password="postgres", autocommit=True) as con, con.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute("""pg_dump {} > {}.sql""".format(db_name, name))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "reset_unfinished":
        reset_unfinished()
        meta = find_unfinished()
    elif len(sys.argv) > 1 and "dump_" in sys.argv[1]:
        dump_name = sys.argv[1].split("_")[1]
        dump_db(dump_name)
    elif len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean_db()
    else:
        meta = get_all_meta()
        total = len(meta)
        completed = len([k for k in meta if k["finished"]])
        remaining = total - completed
        print(json.dumps(get_all_meta(), indent=2))
        print("Total: {}, Completed: {}, Remaining: {}".format(total, completed, remaining))

