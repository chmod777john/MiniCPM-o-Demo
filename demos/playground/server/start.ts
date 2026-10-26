import { createServer } from 'node:http';
import { applicationHandler } from './application.ts';
const host = process.env.PLAYGROUND_APP_HOST || '127.0.0.1';
const port = Number(process.env.PLAYGROUND_APP_PORT || 5187);
const handle = applicationHandler();
createServer((req, res) => {
  void handle(req, res)
    .then((handled) => {
      if (!handled) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ detail: 'Not found' }));
      }
    })
    .catch(() => {
      res.writeHead(500);
      res.end();
    });
}).listen(port, host, () => console.log(`Playground application: http://${host}:${port}`));
