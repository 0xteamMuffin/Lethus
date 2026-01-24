import React, { useState } from "react";
import { User, ChevronDown, ChevronRight, Brain, BarChart3, Zap } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { DYCPStats } from "@/api/message";

interface ChatMessageProps {
  content: string;
  sender: "user" | "ai";
  timestamp?: string;
  isStreaming?: boolean;
  thinking?: string;
  dycpStats?: DYCPStats;
}

const ChatMessage: React.FC<ChatMessageProps> = ({
  content,
  sender,
  timestamp,
  isStreaming,
  thinking,
  dycpStats,
}) => {
  const isUser = sender === "user";
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(false);
  const [isStatsExpanded, setIsStatsExpanded] = useState(false);

  return (
    <div
      className={`w-full flex ${isUser ? "justify-end" : "justify-start"} py-3`}
    >
      <div
        className={`flex gap-3 md:gap-4 max-w-3xl w-full ${isUser ? "flex-row-reverse" : "flex-row"}`}
      >
        <div className="shrink-0 flex flex-col items-center">
          {isUser ? (
            <div className="h-8 w-8 md:h-9 md:w-9 rounded-xl bg-[#1a1a1a] border border-[#2a2a2a] flex items-center justify-center text-gray-300">
              <User size={16} className="md:w-[18px] md:h-[18px]" />
            </div>
          ) : (
            <div className="h-8 w-8 md:h-9 md:w-9 rounded-full bg-white flex items-center justify-center shadow-lg shadow-white/5">
              <span className="text-black text-xs md:text-sm font-bold">L</span>
            </div>
          )}
        </div>

        <div
          className={`flex flex-col max-w-[85%] md:max-w-[85%] ${isUser ? "items-end" : "items-start"}`}
        >
          {!isUser && (
            <span className="text-sm font-semibold text-white mb-2 ml-1">
              Lethus AI
            </span>
          )}

          {/* Thinking/Reasoning section */}
          {!isUser && thinking && (
            <div className="w-full mb-3">
              <button
                onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
                className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-gray-400 hover:text-gray-300 bg-[#151515] border border-[#2a2a2a] rounded-lg transition-colors"
              >
                <Brain size={14} className="text-purple-400" />
                <span>Thinking{isStreaming && thinking ? "..." : ""}</span>
                {isThinkingExpanded ? (
                  <ChevronDown size={14} />
                ) : (
                  <ChevronRight size={14} />
                )}
              </button>
              {isThinkingExpanded && (
                <div className="mt-2 px-4 py-3 bg-[#0d0d0d] border border-[#252525] rounded-lg text-[13px] text-gray-400 leading-relaxed max-h-[300px] overflow-y-auto">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => (
                        <p className="mb-2 last:mb-0">{children}</p>
                      ),
                      code: ({ className, children }) => {
                        const isBlock = className?.includes("language-");
                        return isBlock ? (
                          <pre className="bg-[#111] border border-[#222] rounded-lg p-3 overflow-x-auto my-2 text-xs">
                            <code className={className}>{children}</code>
                          </pre>
                        ) : (
                          <code className="bg-[#222] px-1 py-0.5 rounded text-xs">
                            {children}
                          </code>
                        );
                      },
                    }}
                  >
                    {thinking}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          )}

          {/* DYCP Stats */}
          {!isUser && dycpStats && dycpStats.tokensSaved > 0 && (
            <div className="mt-2 px-3 py-2 bg-[#0d0d0d] border border-[#252525] rounded text-xs text-gray-400">
              <span className="text-emerald-400 font-medium">DYCP:</span> {dycpStats.originalMessages} → {dycpStats.reducedMessages} messages, {dycpStats.originalTokens} → {dycpStats.reducedTokens} tokens ({dycpStats.reductionPercent.toFixed(1)}% reduction, {dycpStats.tokensSaved} tokens saved)
            </div>
          )}

          <div
            className={`px-4 py-3 md:px-5 md:py-3.5 text-[14px] md:text-[15px] leading-6 md:leading-7  ${
              isUser
                ? "bg-[#1a1a1a] border border-[#2a2a2a] text-gray-100 rounded-2xl rounded-tr-md"
                : "bg-transparent text-gray-200 pl-0 pt-0"
            }`}
          >
            {/* Typing indicator when streaming but no content yet */}
            {!isUser && isStreaming && !content ? (
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            ) : (
              <>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => (
                      <p className="mb-2 last:mb-0">{children}</p>
                    ),
                    code: ({ className, children }) => {
                      const isBlock = className?.includes("language-");

                      return isBlock ? (
                        <pre className="bg-[#111] border border-[#2a2a2a] rounded-xl p-4 overflow-x-auto my-3 text-sm">
                          <code className={className}>{children}</code>
                        </pre>
                      ) : (
                        <code className="bg-[#2a2a2a] px-1.5 py-0.5 rounded text-sm">
                          {children}
                        </code>
                      );
                    },
                    a: ({ href, children }) => (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-400 hover:underline"
                      >
                        {children}
                      </a>
                    ),
                    ul: ({ children }) => (
                      <ul className="list-disc pl-6 my-2 space-y-1">{children}</ul>
                    ),
                    ol: ({ children }) => (
                      <ol className="list-decimal pl-6 my-2 space-y-1">
                        {children}
                      </ol>
                    ),
                    h1: ({ children }) => (
                      <h1 className="text-xl font-semibold my-3">{children}</h1>
                    ),
                    h2: ({ children }) => (
                      <h2 className="text-lg font-semibold my-3">{children}</h2>
                    ),
                    h3: ({ children }) => (
                      <h3 className="text-base font-semibold my-2">{children}</h3>
                    ),
                  }}
                >
                  {content}
                </ReactMarkdown>
                {/* Blinking cursor while streaming */}
                {!isUser && isStreaming && content && (
                  <span className="inline-block w-2 h-4 ml-0.5 bg-gray-400 animate-pulse" />
                )}
              </>
            )}
          </div>

          {timestamp && (
            <span
              className={`text-[10px] text-gray-600 mt-1.5 font-mono ${isUser ? "mr-1" : "ml-1"}`}
            >
              {timestamp}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
