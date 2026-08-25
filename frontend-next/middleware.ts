import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// Simple in-memory rate limit map (Note: in edge this is per-isolate, but sufficient for basic protection)
const rateLimit = new Map();

export function middleware(request: NextRequest) {
  if (request.nextUrl.pathname.startsWith('/api/proxy')) {
    // 1. Rate Limiting
    const ip = request.headers.get('x-forwarded-for') || '127.0.0.1';
    const now = Date.now();
    const windowMs = 60 * 1000; // 1 minute
    const maxRequests = 30; // 30 requests per minute
    
    const requestData = rateLimit.get(ip) || { count: 0, startTime: now };
    
    if (now - requestData.startTime > windowMs) {
      requestData.count = 1;
      requestData.startTime = now;
    } else {
      requestData.count++;
    }
    
    rateLimit.set(ip, requestData);
    
    if (requestData.count > maxRequests) {
      fetch(`${request.nextUrl.origin}/api/log`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level: 'WARN', source: 'EdgeMiddleware', message: `Rate limit exceeded for IP: ${ip}` })
      }).catch(() => {});

      return new NextResponse(JSON.stringify({ error: 'Too many requests - Rate limit exceeded' }), {
        status: 429,
        headers: { 'content-type': 'application/json' }
      });
    }

    // 2. Rewrite to Backend
    const backendUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';
    const targetPath = request.nextUrl.pathname.replace('/api/proxy', '');
    const targetUrl = new URL(targetPath, backendUrl);
    
    fetch(`${request.nextUrl.origin}/api/log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level: 'INFO', source: 'EdgeMiddleware', message: `Proxying request to backend: ${targetPath} for IP: ${ip}` })
    }).catch(() => {});

    return NextResponse.rewrite(targetUrl);
  }
}

export const config = {
  matcher: '/api/proxy/:path*',
}
