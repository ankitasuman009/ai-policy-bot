import { Injectable } from '@angular/core';


interface Source {
    document: string;
    page: number;
}

export type StreamEvent =
    | { type: 'token'; content: string }
    | { type: 'replace'; content: string }
    | { type: 'sources'; sources: Source[] }
    | { type: 'done' }
    | { type: 'error'; message: string };


@Injectable({
    providedIn: 'root'
})

export class ChatService {
    private readonly apiUrl = 'http://localhost:8000';
    async streamMessage(
        question: string,
        history: { role: 'user' | 'assistant'; content: string }[],
        onEvent: (event: StreamEvent) => void
    ): Promise<void> {
        const response = await fetch(`${this.apiUrl}/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question, history })
        });

        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }
        if (!response.body) {
            throw new Error('Response stream is not available');
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                break;
            }
            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split(/\r?\n\r?\n/);
            buffer = events.pop() ?? '';
            for (const eventText of events) {
                this.processEvent(eventText, onEvent);
            }
        }
        buffer += decoder.decode();
        if (buffer.trim()) {
            this.processEvent(buffer, onEvent);
        }
    }

    private processEvent(
        eventText: string,
        onEvent: (event: StreamEvent) => void
    ): void {
        let eventType = '';
        let eventData = '';
        const lines = eventText.replace(/\r/g, '').split('\n');
        for (const line of lines) {
            if (line.startsWith('event:')) {
                eventType = line.substring(6).trim();
            }
            if (line.startsWith('data:')) {
                eventData += line.substring(5).trim();
            }
        }
        if (!eventData) {
            return;
        }
        const event = JSON.parse(eventData) as StreamEvent;
        onEvent({
            ...event,
            type: event.type || eventType
        } as StreamEvent);
    }
}
