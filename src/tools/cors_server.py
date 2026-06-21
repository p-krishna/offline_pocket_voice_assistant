from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Allow access from any origin
        self.send_header('Access-Control-Allow-Origin', '*')
        # Allow standard HTTP methods
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        # Allow common headers
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        # Reply to CORS preflight requests
        self.send_response(200, "OK")
        self.end_headers()

if __name__ == '__main__':
    # Define your target directory here
    target_dir = '/tmp'
    
    # Ensure the directory exists before serving
    os.makedirs(target_dir, exist_ok=True)
    
    # Bind the directory path to the handler class
    handler = partial(CORSRequestHandler, directory=target_dir)
    
    print(f"Starting CORS-enabled server serving: {target_dir}")
    print("Listening on http://localhost:9999")
    
    server = HTTPServer(('localhost', 9999), handler)
    server.serve_forever()
