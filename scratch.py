import requests

with open("test.pdf", "wb") as f:
    f.write(b"%PDF-1.4 dummy")

url = "https://tmpfiles.org/api/v1/upload"
with open("test.pdf", 'rb') as f:
    response = requests.post(url, files={'file': f})
    
print("Status Code:", response.status_code)
print("Text:", response.text)
try:
    data = response.json()
    url = data['data']['url']
    print("URL:", url)
    # The API returns https://tmpfiles.org/1234/test.pdf
    # The direct link is https://tmpfiles.org/dl/1234/test.pdf
    direct_url = url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
    print("Direct URL:", direct_url)
except Exception as e:
    print("JSON Parse Error:", e)
