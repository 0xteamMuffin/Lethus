import { apiFetch } from "./http";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface SendMessageParams {
  messages: ChatMessage[];
  userId: string;
  model?: string;
  stream?: boolean;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface DYCPStats {
  originalMessages: number;
  reducedMessages: number;
  originalTokens: number;
  reducedTokens: number;
  tokensSaved: number;
  reductionPercent: number;
}

export interface Conversation {
  id: number;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Turn {
  id: number;
  conversation_id: number;
  turn_number: number;
  user_message: string;
  assistant_message: string;
  timestamp: string;
  importance_score: number;
  is_pinned: boolean;
}

export interface ConversationWithTurns extends Conversation {
  turns: Turn[];
}

/**
 * Send chat completion request via proxy.
 * Uses user's stored API key from database.
 */
export async function sendChatCompletion(params: SendMessageParams): Promise<{
  response: ChatCompletionResponse;
  dycpStats: DYCPStats;
}> {
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: params.model || "gpt-4o-mini",
      messages: params.messages,
      user_id: params.userId,
      stream: false,
    }),
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Chat request failed");
  }

  // Extract DYCP stats from headers
  const dycpStats: DYCPStats = {
    originalMessages: parseInt(res.headers.get("X-Lethus-Original-Messages") || "0"),
    reducedMessages: parseInt(res.headers.get("X-Lethus-Reduced-Messages") || "0"),
    originalTokens: parseInt(res.headers.get("X-Lethus-Original-Tokens") || "0"),
    reducedTokens: parseInt(res.headers.get("X-Lethus-Reduced-Tokens") || "0"),
    tokensSaved: parseInt(res.headers.get("X-Lethus-Tokens-Saved") || "0"),
    reductionPercent: parseFloat(res.headers.get("X-Lethus-Reduction-Percent") || "0"),
  };

  const response = await res.json();
  return { response, dycpStats };
}

/**
 * Stream chat completion via proxy.
 * Uses user's stored API key from database.
 */
export async function streamChatCompletion(
  params: SendMessageParams,
  onChunk: (content: string) => void,
  onComplete?: (dycpStats: DYCPStats) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: params.model || "gpt-4o-mini",
      messages: params.messages,
      user_id: params.userId,
      stream: true,
    }),
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Chat request failed");
  }

  // Extract DYCP stats from headers
  const dycpStats: DYCPStats = {
    originalMessages: parseInt(res.headers.get("X-Lethus-Original-Messages") || "0"),
    reducedMessages: parseInt(res.headers.get("X-Lethus-Reduced-Messages") || "0"),
    originalTokens: parseInt(res.headers.get("X-Lethus-Original-Tokens") || "0"),
    reducedTokens: parseInt(res.headers.get("X-Lethus-Reduced-Tokens") || "0"),
    tokensSaved: parseInt(res.headers.get("X-Lethus-Tokens-Saved") || "0"),
    reductionPercent: parseFloat(res.headers.get("X-Lethus-Reduction-Percent") || "0"),
  };

  const reader = res.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error("No response body");
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split("\n");

    for (const line of lines) {
      if (line.startsWith("data: ") && line !== "data: [DONE]") {
        try {
          const data = JSON.parse(line.slice(6));
          const content = data.choices?.[0]?.delta?.content;
          if (content) {
            onChunk(content);
          }
        } catch {
          // Skip invalid JSON
        }
      }
    }
  }

  onComplete?.(dycpStats);
}

export async function createConversation(userId: string, title?: string): Promise<Conversation> {
  return apiFetch("/api/conversations", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      title: title || "New Conversation",
    }),
  });
}

export async function getConversations(userId: string): Promise<Conversation[]> {
  return apiFetch(`/api/conversations/user/${userId}`);
}

export async function getConversation(conversationId: number): Promise<Conversation> {
  return apiFetch(`/api/conversations/${conversationId}`);
}

export async function getConversationTurns(conversationId: number): Promise<Turn[]> {
  return apiFetch(`/api/conversations/${conversationId}/turns`);
}

export async function deleteConversation(conversationId: number): Promise<void> {
  return apiFetch(`/api/conversations/${conversationId}`, {
    method: "DELETE",
  });
}

export async function getMemoryStats() {
  return apiFetch("/api/stats");
}
