import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dashboard.app import app


def test_api_data():
    client = app.test_client()
    resp = client.get('/api/data?country=Afghanistan&start=2020-03-01&end=2020-03-10')
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert isinstance(data, list)
    assert len(data) > 0
