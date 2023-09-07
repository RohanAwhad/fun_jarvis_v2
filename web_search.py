from duckpy import Client


def search(txt):
  client = Client()
  search_results = client.search(txt)
  return [x.url for x in search_results]
