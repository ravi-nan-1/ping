// api/ping.js
import fetch from "node-fetch";

export default async function handler(req, res) {
  const urls = process.env.PING_URLS?.split(",") || [];
  const results = [];

  for (const url of urls) {
    try {
      const r = await fetch(url.trim());
      results.push({ url, status: r.status });
    } catch (error) {
      results.push({ url, status: "ERROR", error: error.message });
    }
  }

  return res.status(200).json({ ok: true, results });
}
