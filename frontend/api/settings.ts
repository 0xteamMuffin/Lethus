import { apiFetch } from "./http";

export interface UserSettings {
  user_id: string;
  has_api_key: boolean;
  llm_model: string | null;
  embedding_model: string | null;
  default_llm_model: string;
  default_embedding_model: string;
  created_at: string;
  updated_at: string;
}

export interface ValidateApiKeyResponse {
  valid: boolean;
  error: string | null;
}

export interface SaveSettingsParams {
  userId: string;
  openaiApiKey?: string;
  llmModel?: string;
  embeddingModel?: string;
}

export async function saveUserSettings(params: SaveSettingsParams): Promise<UserSettings> {
  const body: Record<string, string | undefined> = {
    user_id: params.userId,
  };
  
  if (params.openaiApiKey !== undefined) {
    body.openai_api_key = params.openaiApiKey;
  }
  if (params.llmModel !== undefined) {
    body.llm_model = params.llmModel;
  }
  if (params.embeddingModel !== undefined) {
    body.embedding_model = params.embeddingModel;
  }
  
  return apiFetch("/api/settings", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getUserSettings(userId: string): Promise<UserSettings> {
  return apiFetch(`/api/settings/${userId}`);
}

export async function validateApiKey(apiKey: string): Promise<ValidateApiKeyResponse> {
  return apiFetch("/api/settings/validate-api-key", {
    method: "POST",
    body: JSON.stringify({ api_key: apiKey }),
  });
}

export async function deleteApiKey(userId: string): Promise<void> {
  return apiFetch(`/api/settings/${userId}/api-key`, {
    method: "DELETE",
  });
}

export interface AvailableModels {
  llm_models: string[];
  embedding_models: string[];
}

export async function getAvailableModels(userId: string): Promise<AvailableModels> {
  return apiFetch(`/api/settings/${userId}/models`);
}
