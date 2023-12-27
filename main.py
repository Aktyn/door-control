import json
from time import sleep
from motor_control import Direction, MotorControl
from http.server import BaseHTTPRequestHandler, HTTPServer


class ServerHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self._motor = MotorControl()
        super().__init__(*args, **kwargs)

    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.end_headers()

    def do_GET(self):
        print(
            "GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers)
        )
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        decoded_data = post_data.decode("utf-8")
        print(
            "POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
            str(self.path),
            str(self.headers),
            decoded_data,
        )

        self.handle_JSON_body(decoded_data)

        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode("utf-8"))

    def handle_JSON_body(self, body: str):
        data = json.loads(body)
        
        if not "action" in data:
            return
        
        action = data["action"]
        if 'duration' in data:
            duration = data["duration"]
        else:
            duration = 2

        if self._motor is None:
            self._motor = MotorControl()

        if action == "open":
            self._motor.run(direction=Direction.FORWARD)
            sleep(duration)
            self._motor.stop()
            pass
        elif action == "close":
            self._motor.run(direction=Direction.BACKWARD)
            sleep(duration)
            self._motor.stop()
            pass


def main():
    print("Starting door control")

    server_address = ("", 7331)
    httpd = HTTPServer(server_address, ServerHandler)
    print("Starting httpd")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Stopping httpd")


if __name__ == "__main__":
    main()
