import requests, json, sys

r = requests.get(
    'https://nominatim.openstreetmap.org/search',
    params={'q': 'Bhote Koshi Nepal', 'format': 'json', 'limit': 3},
    headers={'User-Agent': 'FloodMapper/1.0'},
    timeout=10
)
sys.stdout.reconfigure(encoding='utf-8')
data = r.json()
for d in data[:3]:
    print(json.dumps({
        'name': d.get('display_name', '')[:80],
        'lat': d['lat'], 'lon': d['lon'],
        'bbox': d.get('boundingbox')
    }, ensure_ascii=False))
