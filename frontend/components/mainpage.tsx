"use client";
import React, { useState, useRef, useEffect } from "react";
import {
  ChevronDown,
  Paperclip,
  Code,
  Server,
  Mic,
  ArrowUp,
  LayoutGrid,
  Menu,
  Radar,
  Settings,
} from "lucide-react";
import { toast, Toaster } from "sonner";
import {
  streamChatCompletion,
  getConversationTurns,
  createConversation,
  type Turn,
  type ChatMessage as APIChatMessage,
  type DYCPStats,
} from "@/api/message";
import { getUserSettings } from "@/api/settings";
import ChatMessageComponent from "./ui/chat-message";
import SettingsModal from "./ui/settings-modal";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface LibreChatInterfaceProps {
  onToggleSidebar?: () => void;
  onToggleMemory?: () => void;
  conversationId?: number;
  onConversationCreated?: (id: number) => void;
}

interface Message {
  id: string;
  content: string;
  sender: "user" | "ai";
  timestamp: string;
}

const LibreChatInterface: React.FC<LibreChatInterfaceProps> = ({
  onToggleSidebar,
  onToggleMemory,
  conversationId: propConversationId,
  onConversationCreated,
}) => {
  const [message, setMessage] = useState<string>("");
  const [isSending, setIsSending] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatHistory, setChatHistory] = useState<APIChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<number | undefined>(
    propConversationId,
  );
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [apiKey, setApiKey] = useState<string>("");
  const [hasApiKey, setHasApiKey] = useState<boolean>(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [lastDycpStats, setLastDycpStats] = useState<DYCPStats | null>(null);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(
    null,
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const streamingMessageRef = useRef<string>("");

  // Check if user has API key in backend on mount
  useEffect(() => {
    const checkApiKey = async () => {
      if (typeof window === "undefined") return;

      try {
        const userId = getUserId();
        if (!userId) return;

        const settings = await getUserSettings(userId);
        setHasApiKey(settings.has_api_key);
      } catch (error) {
        console.error("Failed to check API key:", error);
      }
    };

    checkApiKey();
  }, []);

  // Update conversation ID when prop changes and load history
  useEffect(() => {
    if (propConversationId !== conversationId) {
      setConversationId(propConversationId);
      if (propConversationId) {
        loadConversationHistory(propConversationId);
      } else {
        setMessages([]);
        setChatHistory([]);
      }
    }
  }, [propConversationId]);

  // Load conversation history
  const loadConversationHistory = async (convId: number) => {
    try {
      setIsLoadingHistory(true);
      const turns = await getConversationTurns(convId);

      const loadedMessages: Message[] = [];
      const loadedChatHistory: APIChatMessage[] = [];

      turns.forEach((turn: Turn) => {
        loadedMessages.push({
          id: `${turn.id}-user`,
          content: turn.user_message,
          sender: "user",
          timestamp: new Date(turn.timestamp).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
        });
        loadedMessages.push({
          id: `${turn.id}-ai`,
          content: turn.assistant_message,
          sender: "ai",
          timestamp: new Date(turn.timestamp).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
        });

        loadedChatHistory.push({ role: "user", content: turn.user_message });
        loadedChatHistory.push({
          role: "assistant",
          content: turn.assistant_message,
        });
      });

      setMessages(loadedMessages);
      setChatHistory(loadedChatHistory);
    } catch (error) {
      console.error("Failed to load conversation history:", error);
      toast.error("Failed to load conversation history");
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const getUserId = (): string => {
    if (typeof window === "undefined") {
      return "";
    }
    let userId = localStorage.getItem("user_id");
    if (!userId) {
      userId = `user_${Date.now()}_${Math.random().toString(36).substring(7)}`;
      localStorage.setItem("user_id", userId);
    }
    return userId;
  };

  const handleSaveApiKey = (newApiKey: string) => {
    setApiKey(newApiKey);
    setHasApiKey(true);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  // Store turn in database
  const storeTurn = async (
    convId: number,
    turnNumber: number,
    userMsg: string,
    assistantMsg: string,
  ) => {
    try {
      await fetch(`${API_BASE}/api/conversations/${convId}/turns`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          turn_number: turnNumber,
          user_message: userMsg,
          assistant_message: assistantMsg,
        }),
      });
    } catch (error) {
      console.error("Failed to store turn:", error);
    }
  };

  const handleSendMessage = async () => {
    if (!message.trim() || isSending) return;

    if (!hasApiKey) {
      toast.error("Please configure your OpenAI API key in settings");
      setIsSettingsOpen(true);
      return;
    }

    setIsSending(true);
    streamingMessageRef.current = "";

    const userMessage: Message = {
      id: Date.now().toString(),
      content: message,
      sender: "user",
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };

    const aiMessageId = (Date.now() + 1).toString();
    const aiMessage: Message = {
      id: aiMessageId,
      content: "",
      sender: "ai",
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };

    setMessages((prev) => [...prev, userMessage, aiMessage]);
    setStreamingMessageId(aiMessageId);

    const newUserMessage: APIChatMessage = { role: "user", content: message };
    const updatedHistory = [...chatHistory, newUserMessage];
    setChatHistory(updatedHistory);

    const messageToSend = message;
    setMessage("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      // Create conversation if this is the first message
      let currentConvId = conversationId;
      if (!currentConvId) {
        const conv = await createConversation(
          getUserId(),
          messageToSend.slice(0, 50),
        );
        currentConvId = conv.id;
        setConversationId(currentConvId);
        onConversationCreated?.(currentConvId);
      }

      await streamChatCompletion(
        {
          messages: updatedHistory,
          userId: getUserId(),
          model: "gpt-4o-mini",
          stream: true,
        },
        (chunk) => {
          streamingMessageRef.current += chunk;
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId
                ? { ...msg, content: streamingMessageRef.current }
                : msg,
            ),
          );
        },
        (stats) => {
          setStreamingMessageId(null);
          setLastDycpStats(stats);
          if (stats.tokensSaved > 0) {
            toast.success(
              `DYCP saved ~${stats.tokensSaved.toLocaleString()} tokens`,
            );
          }
        },
      );

      // Update chat history with assistant response
      const assistantMessage: APIChatMessage = {
        role: "assistant",
        content: streamingMessageRef.current,
      };
      setChatHistory([...updatedHistory, assistantMessage]);

      // Store turn in database
      const turnNumber = Math.floor(chatHistory.length / 2) + 1;
      await storeTurn(
        currentConvId!,
        turnNumber,
        messageToSend,
        streamingMessageRef.current,
      );
    } catch (error) {
      console.error(error);
      const errorMsg =
        error instanceof Error ? error.message : "Failed to get response";
      toast.error(errorMsg);

      // Remove the empty AI message on error
      setMessages((prev) => prev.filter((msg) => msg.id !== aiMessageId));
      setChatHistory(updatedHistory.slice(0, -1));
    } finally {
      setIsSending(false);
      setStreamingMessageId(null);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex-1 h-full bg-[#0a0a0a] text-gray-100 flex flex-col relative overflow-hidden w-full">
      <Toaster position="top-center" theme="dark" />{" "}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSave={handleSaveApiKey}
        currentApiKey={apiKey}
        userId={getUserId()}
      />
      <header className="flex justify-between items-center px-4 md:px-6 py-3 md:py-4 text-gray-400 border-b border-[#2a2a2a] bg-[#0a0a0a] z-10 shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleSidebar}
            className="p-2 -ml-2 text-gray-400 hover:text-white md:hidden"
          >
            <Menu size={20} />
          </button>

          <div className="flex items-center gap-2 hover:bg-[#1a1a1a] px-3 md:px-4 py-2 rounded-xl cursor-pointer transition-all duration-200 text-sm md:text-base font-medium text-gray-200">
            <span>Lethus AI</span>
            <ChevronDown size={16} className="text-gray-500" />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsSettingsOpen(true)}
            className="p-2 rounded-lg hover:bg-[#1a1a1a] hover:text-white transition-all duration-200"
            title="Settings"
          >
            <Settings size={18} />
          </button>
          <button className="p-2 rounded-lg hover:bg-[#1a1a1a] hover:text-white transition-all duration-200">
            <LayoutGrid size={18} />
          </button>

          <button
            onClick={onToggleMemory}
            className="p-2 text-gray-400 hover:text-white lg:hidden"
          >
            <Radar size={18} />
          </button>
        </div>
      </header>
      <main className="flex-1 flex flex-col w-full overflow-y-auto min-h-0">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full px-4">
            <div className="text-center max-w-2xl animate-fade-in">
              <div className="bg-white text-black p-3 rounded-full inline-flex items-center justify-center mb-6 shadow-lg shadow-white/5">
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  className="text-black"
                >
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-5-9h10v2H7z" />
                </svg>
              </div>
              <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-white mb-3">
                Lethus AI
              </h1>
              <p className="text-sm md:text-base text-gray-500 max-w-md mx-auto leading-relaxed px-4 mb-6">
                Context-aware AI assistant with dynamic memory management
              </p>

              {!hasApiKey && (
                <div className="mt-8 p-4 bg-[#121212] border border-[#2a2a2a] rounded-xl max-w-md mx-auto">
                  <div className="flex items-start gap-3 mb-3">
                    <Settings
                      size={18}
                      className="text-gray-400 mt-0.5 shrink-0"
                    />
                    <div className="text-left">
                      <h3 className="text-sm font-semibold text-white mb-1">
                        API Key Required
                      </h3>
                      <p className="text-xs text-gray-500">
                        Configure your OpenAI API key in settings to start
                        chatting
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setIsSettingsOpen(true)}
                    className="w-full px-4 py-2 bg-white text-black rounded-lg text-sm font-medium hover:bg-gray-200 transition-all"
                  >
                    Configure API Key
                  </button>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="flex flex-col w-full items-center p-2 md:p-4 pb-0">
            {messages.map((msg) => (
              <ChatMessageComponent
                key={msg.id}
                content={msg.content}
                sender={msg.sender}
                timestamp={msg.timestamp}
                isStreaming={msg.id === streamingMessageId}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>
      <footer className="w-full px-4 md:px-6 py-4 md:py-6 border-t border-[#2a2a2a] bg-[#0a0a0a] shrink-0">
        <div className="flex justify-center w-full">
          <div className="w-full max-w-3xl bg-[#121212] rounded-2xl p-3 md:p-4">
            <div className="mb-2 md:mb-3 px-2">
              <textarea
                ref={textareaRef}
                value={message}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                disabled={isSending}
                placeholder={isSending ? "Thinking..." : "Message Lethus AI"}
                rows={1}
                className="w-full bg-transparent text-white placeholder-gray-500 resize-none overflow-y-auto max-h-[150px] md:max-h-[200px] scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent disabled:cursor-not-allowed disabled:opacity-50 text-sm md:text-base"
                style={{
                  minHeight: "24px",
                  outline: "none",
                  border: "none",
                  boxShadow: "none",
                }}
              />
            </div>

            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-0 mt-2 md:mt-3">
              <div className="flex items-center gap-2 overflow-x-auto w-full sm:w-auto pb-1 sm:pb-0 scrollbar-none">
                <button className="p-2 text-gray-400 hover:text-white hover:bg-[#1a1a1a] rounded-lg transition-all duration-200 shrink-0">
                  <Paperclip size={18} />
                </button>

                <button className="flex items-center gap-2 bg-[#1a1a1a] px-3 py-2 rounded-lg border border-[#2a2a2a] hover:bg-[#212121] hover:border-[#333333] transition-all duration-200 text-xs font-medium text-gray-300 shrink-0">
                  <Code size={14} />
                  <span>Code</span>
                </button>

                <button className="flex items-center gap-2 bg-[#1a1a1a] px-3 py-2 rounded-lg border border-[#2a2a2a] hover:bg-[#212121] hover:border-[#333333] transition-all duration-200 text-xs font-medium text-gray-300 shrink-0">
                  <Server size={14} />
                  <span>MCP</span>
                  <ChevronDown size={12} className="opacity-70" />
                </button>
              </div>

              <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
                <button className="p-2 text-gray-400 hover:text-white hover:bg-[#1a1a1a] rounded-lg transition-all duration-200">
                  <Mic size={20} />
                </button>

                <button
                  onClick={handleSendMessage}
                  disabled={!message.trim() || isSending}
                  className={`p-2.5 rounded-lg transition-all duration-200 ${
                    message.trim() && !isSending
                      ? "bg-white text-black hover:bg-gray-200 shadow-lg shadow-white/10"
                      : "bg-[#2a2a2a] text-gray-600 cursor-not-allowed"
                  }`}
                >
                  <ArrowUp size={18} strokeWidth={2.5} />
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="w-full text-center mt-3 text-[10px] md:text-[11px] text-gray-600 hidden sm:block">
          <p>
            Lethus AI uses dynamic context engineering. Verify important
            information.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LibreChatInterface;
