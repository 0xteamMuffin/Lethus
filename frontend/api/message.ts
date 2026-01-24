import { apiFetch } from "./http";

export interface SendMessageParams {
  message: string;
  userId: string;
  conversationId?: number;
  openaiApiKey?: string;  // Deprecated - API key now stored in backend
}

export interface MessageResponse {
  conversation_id: number;
  turn_id: number;
  message: string;
  retrieved_context: {
    spans: Array<{
      start_index: number;
      end_index: number;
      turns: Array<{ role: string; content: string }>;
      relevance_score: number;
    }>;
    pinned_memories: Array<{ content: string; importance_score: number }>;
    confidence: { confident: boolean; score: number };
  };
  metadata: {
    turn_number: number;
    entities: string[];
  };
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

export async function sendMessage(params: SendMessageParams): Promise<MessageResponse> {
  const body: any = {
    message: params.message,
    user_id: params.userId,
    conversation_id: params.conversationId,
  };
  
  // Include API key only if provided (for backward compatibility)
  if (params.openaiApiKey) {
    body.openai_api_key = params.openaiApiKey;
  }
  
  return apiFetch("/api/chat", {
    method: "POST",
    body: JSON.stringify(body),
  });
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
