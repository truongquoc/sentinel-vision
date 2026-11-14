import asyncio, os, threading, time
import cv2

class CameraReader:
    def __init__(self, camera_id, url):
        self.camera_id=camera_id; self.url=url; self.frame=None; self.annotated_frame=None; self.online=False; self.last_frame_at=None; self.last_error=None; self._stop=threading.Event(); self._thread=None
    def start(self):
        if self._thread and self._thread.is_alive(): return
        self._stop.clear(); self._thread=threading.Thread(target=self._run,name=f"camera-{self.camera_id}",daemon=True); self._thread.start()
    def stop(self): self._stop.set()
    def _run(self):
        backoff=1
        while not self._stop.is_set():
            if not self.url:
                self.online=False; self.last_error="No RTSP URL configured"; self._stop.wait(10); continue
            # TCP is generally more reliable than UDP across Wi-Fi and VLANs.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            cap=cv2.VideoCapture(self.url,cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.online=False; self.last_error="Unable to open RTSP stream (check camera reachability, credentials, RTSP settings, and path)"; self._stop.wait(backoff); backoff=min(backoff*2,30); continue
            backoff=1
            while not self._stop.is_set():
                ok, frame=cap.read()
                if not ok:
                    self.online=False; self.last_error="RTSP connection opened but no video frames arrived"; break
                self.frame=frame; self.online=True; self.last_error=None; self.last_frame_at=time.time()
            cap.release()
    def jpeg(self, annotated=True):
        image = self.annotated_frame if annotated and self.annotated_frame is not None else self.frame
        if image is None: return None
        ok, data=cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY,80]); return data.tobytes() if ok else None
    async def mjpeg(self):
        while True:
            image=self.jpeg()
            if image: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+image+b"\r\n"
            await asyncio.sleep(.08)
