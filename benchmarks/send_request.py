import requests, PIL.Image as Image, io, base64, time

url = "https://xskcq-01gmcqx8r148q461b5gwr5cmgt.litng-ai-03.litng.ai/predict"
response = requests.post(url, json={
  "text": "Woman painting a large red egg in a dali landscape"
})

print(response.json())

image = Image.open(io.BytesIO(base64.b64decode(response.json()["image"])))
image.save(f"response_{time.time()}.png")
