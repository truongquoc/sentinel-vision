import hashlib, hmac, json
import httpx

class WebhookNotifier:
    def __init__(self,url,secret): self.url=url; self.secret=secret
    async def send(self,event):
        if not self.url: return
        body=json.dumps(event,default=str).encode(); headers={"Content-Type":"application/json"}
        if self.secret: headers["X-Sentinel-Signature"]="sha256="+hmac.new(self.secret.encode(),body,hashlib.sha256).hexdigest()
        async with httpx.AsyncClient(timeout=5) as client: await client.post(self.url,content=body,headers=headers)
