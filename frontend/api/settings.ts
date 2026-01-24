import { apiFetch } from "./http";

export interface UserSettings {
  user_id: string;
  has_api_key: boolean;
  created_at: string;
  updated_at: string;
}

export interface SaveSettingsParams {
  userId: string;
  openaiApiKey: string;
}

export async function saveUserSettings(params: SaveSettingsParams): Promise<UserSettings> {
  return apiFetch("/api/settings", {
    method: "POST",
    body: JSON.stringify({
      user_id: params.userId,
      openai_api_key: params.openaiApiKey,
    }),
  });
}

export async function getUserSettings(userId: string): Promise<UserSettings> {
  return apiFetch(`/api/settings/${userId}`);
}
