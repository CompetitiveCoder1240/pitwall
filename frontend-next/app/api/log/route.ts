import { NextRequest, NextResponse } from 'next/server';
import { logToServer } from '@/utils/logger';

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const { level = 'INFO', source = 'Unknown', message = '' } = body;
        
        logToServer(level, source, message);
        
        return NextResponse.json({ success: true });
    } catch (e) {
        return NextResponse.json({ error: 'Invalid payload' }, { status: 400 });
    }
}
