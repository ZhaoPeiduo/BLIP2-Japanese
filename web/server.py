import io
import json
import os
import posixpath
import sys
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote

from PIL import Image
import torch

# Append sys.path to allow relative imports when run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazily import heavy deps so the module can be imported fast
from lavis.models import load_model_and_preprocess


# Load the model once at startup
print("Loading BLIP-2 Japanese model…", flush=True)
MODEL, VIS_PREPROCESS, TEXT_PREPROCESS = load_model_and_preprocess('blip2_Japanese', 'finetune')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    MODEL = MODEL.to(DEVICE)
except Exception:
    # Some models manage device internally; ignore if unsupported
    pass
print(f"Model ready on {DEVICE}", flush=True)


class AppHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        # Serve files from repo root, but default to /web/ for static assets
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        trailing_slash = path.rstrip().endswith('/')
        try:
            path = unquote(path, errors='surrogatepass')
        except Exception:
            path = unquote(path)
        path = posixpath.normpath(path)
        parts = [p for p in path.split('/') if p]

        base = os.getcwd()

        # Map "/" to web/index.html
        if len(parts) == 0:
            return os.path.join(base, 'web', 'index.html')

        # Allow direct access to files under /web
        if parts and parts[0] == 'web':
            return os.path.join(base, *parts)

        # Fallback to web path for unknown top-level GETs (e.g., "/app.js")
        return os.path.join(base, 'web', *parts)

    def do_GET(self):
        # Let SimpleHTTPRequestHandler serve static files
        if self.path.startswith('/api/'):
            self.send_error(HTTPStatus.NOT_FOUND, "No such endpoint")
            return
        return super().do_GET()

    def do_POST(self):
        if self.path != '/api/caption':
            self.send_error(HTTPStatus.NOT_FOUND, "No such endpoint")
            return

        # Parse multipart form without external deps
        ctype = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in ctype:
            self.respond_json({"error": "Expected multipart/form-data"}, status=HTTPStatus.BAD_REQUEST)
            return

        boundary = None
        for part in ctype.split(';'):
            part = part.strip()
            if part.startswith('boundary='):
                boundary = part.split('=', 1)[1]
                if boundary.startswith('"') and boundary.endswith('"'):
                    boundary = boundary[1:-1]
                break
        if not boundary:
            self.respond_json({"error": "Missing multipart boundary"}, status=HTTPStatus.BAD_REQUEST)
            return

        length = int(self.headers.get('Content-Length', '0') or '0')
        body = self.rfile.read(length)

        # Very small multipart parser for single file field named 'image'
        delimiter = ('--' + boundary).encode('utf-8')
        parts = body.split(delimiter)
        file_bytes = None
        for p in parts:
            # skip preamble and closing
            if not p or p in (b'--\r\n', b'--'):
                continue
            # Separate headers and data
            try:
                head, data = p.split(b"\r\n\r\n", 1)
            except ValueError:
                continue
            headers = head.decode('utf-8', errors='ignore')
            if 'name="image"' in headers:
                # Trim trailing CRLF and boundary end markers
                if data.endswith(b"\r\n"):
                    data = data[:-2]
                if data.endswith(b"--"):
                    data = data[:-2]
                file_bytes = data
                break

        if not file_bytes:
            self.respond_json({"error": "Field 'image' not found"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        except Exception as e:
            self.respond_json({"error": f"Invalid image: {e}"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            # Preprocess and run model
            with torch.no_grad():
                image_t = VIS_PREPROCESS['eval'](image)
                image_t = torch.unsqueeze(image_t, 0).to(DEVICE)
                inputs = {"image": image_t}
                caption = MODEL.generate(inputs)[0]
        except Exception as e:
            self.respond_json({"error": f"Model error: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.respond_json({"caption": caption})

    def respond_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '7860'))
    httpd = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Serving on http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…", flush=True)
    finally:
        httpd.server_close()


if __name__ == '__main__':
    main()
