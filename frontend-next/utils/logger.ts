import fs from 'fs';
import path from 'path';

// This file must ONLY be imported in Node.js environments (Server Components, API Routes)
// It cannot be used in middleware.ts (Edge Runtime)

const LOG_FILE_PATH = path.join(process.cwd(), '..', 'logs', 'frontend.log');

// Ensure directory exists
const dir = path.dirname(LOG_FILE_PATH);
if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
}

export function logToServer(level: 'INFO' | 'WARN' | 'ERROR', source: string, message: string) {
    const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19);
    const logLine = `${timestamp} | ${level.padEnd(8)} | ${source} - ${message}\n`;
    
    // Write to console for Vercel/Docker stdout logs
    if (level === 'ERROR') {
        console.error(logLine.trim());
    } else if (level === 'WARN') {
        console.warn(logLine.trim());
    } else {
        console.log(logLine.trim());
    }

    // Append to local file
    try {
        fs.appendFileSync(LOG_FILE_PATH, logLine);
    } catch (e) {
        console.error("Failed to write to local frontend log file:", e);
    }
}
