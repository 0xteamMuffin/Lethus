import React from "react";
import { User } from "lucide-react";

interface ChatMessageProps {
  content: string;
  sender: "user" | "ai";
  timestamp?: string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ content, sender, timestamp }) => {
  const isUser = sender === "user";

  return (
    <div className={`w-full flex ${isUser ? "justify-end" : "justify-start"} py-3`}>
      <div className={`flex gap-3 md:gap-4 max-w-3xl w-full ${isUser ? "flex-row-reverse" : "flex-row"}`}>
        
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

        <div className={`flex flex-col max-w-[85%] md:max-w-[85%] ${isUser ? "items-end" : "items-start"}`}>
            
            {!isUser && (
                <span className="text-sm font-semibold text-white mb-2 ml-1">Lethus AI</span>
            )}

            <div
                className={`px-4 py-3 md:px-5 md:py-3.5 text-[14px] md:text-[15px] leading-6 md:leading-7 whitespace-pre-wrap ${
                isUser
                    ? "bg-[#1a1a1a] border border-[#2a2a2a] text-gray-100 rounded-2xl rounded-tr-md"
                    : "bg-transparent text-gray-200 pl-0 pt-0"
                }`}
            >
                <p>{content}</p>
            </div>

            {timestamp && (
                <span className={`text-[10px] text-gray-600 mt-1.5 font-mono ${isUser ? "mr-1" : "ml-1"}`}>
                    {timestamp}
                </span>
            )}
        </div>

      </div>
    </div>
  );
};

export default ChatMessage;