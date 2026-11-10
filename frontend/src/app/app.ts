import { Component, signal, inject, ElementRef, viewChild, afterRenderEffect } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatService, StreamEvent } from './chat.service';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: { document: string; page: number }[];
  isError?: boolean;
  error?: string;
}

@Component({
  imports: [FormsModule],
  selector: 'app-root',
  styleUrl: './app.css',
  templateUrl: './app.html',
})
export class App {

  private chatService = inject(ChatService);

  protected question = '';

  protected messages = signal<Message[]>([]);

  protected isLoading = signal(false);

  protected async sendMessage() {

    const question = this.question.trim();
    const history = this.messages().map(message => ({
      role: message.role,
      content: message.content
    }));

    if (!question || this.isLoading()) {
      return;
    }

    this.question = '';
    this.isLoading.set(true);

    // Add the user's question.
    this.messages.update(current => [
      ...current,
      { role: 'user', content: question }
    ]);

    // Create an empty AI message.
    this.messages.update(current => [
      ...current,
      { role: 'assistant', content: '' }
    ]);

    const assistantIndex = this.messages().length - 1;

    try {
      await this.chatService.streamMessage(
        question,
        history,
        (event: StreamEvent) => {
          if (event.type === 'token') {
            this.messages.update(current =>
              current.map((message, index) =>
                index === assistantIndex
                  ? { ...message, content: (message.content + event.content).replace(/^\s+/, '') }
                  : message
              )
            );
          } else if (event.type === 'replace') {
            this.messages.update(current =>
              current.map((message, index) =>
                index === assistantIndex
                  ? { ...message, content: event.content }
                  : message
              )
            );
          } else if (event.type === 'sources') {
            this.messages.update(current =>
              current.map((message, index) =>
                index === assistantIndex
                  ? { ...message, sources: event.sources }
                  : message
              )
            );
          } else if (event.type === 'error') {
            this.messages.update(current =>
              current.map((message, index) =>
                index === assistantIndex
                  ? message.content
                    ? {
                      ...message,
                      error: event.message
                    }
                    : {
                      ...message,
                      content: event.message,
                      isError: true
                    }
                  : message
              )
            );
          }
        }
      );
    } catch (error) {
      console.error('Chat error:', error);
      this.messages.update(current =>
        current.map((message, index) =>
          index === assistantIndex
            ? {
              ...message,
              content: 'Unable to connect to the AI assistant. Please try again.',
              isError: true
            }
            : message
        )
      );
    } finally {
      this.isLoading.set(false);
    }
  }

  private messageContainer =
    viewChild<ElementRef<HTMLDivElement>>('messageContainer');

  constructor() {
    afterRenderEffect(() => {
      this.messages();
      const container = this.messageContainer()?.nativeElement;
      if (container) {
        container.scrollTop = container.scrollHeight;
      }
    });
  }
}
