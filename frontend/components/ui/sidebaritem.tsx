import React from 'react';
import { LucideIcon } from 'lucide-react';

interface SidebarItemProps {
  label: string;
  icon: LucideIcon;
  isActive?: boolean;
  onClick?: () => void;
  isBold?: boolean;
}

const SidebarItem: React.FC<SidebarItemProps> = ({ 
  label, 
  icon: Icon, 
  isActive = false, 
  onClick,
  isBold = false
}) => {
  return (
    <button
      onClick={onClick}
      className={`
        group flex w-full cursor-pointer items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-all duration-200 ease-in-out
        ${isActive 
          ? 'bg-[#1a1a1a] text-white border border-[#2a2a2a]' 
          : 'text-gray-400 hover:bg-[#121212] hover:text-gray-100'
        }
      `}
    >
      <Icon 
        size={16} 
        strokeWidth={2}
        className={`shrink-0 transition-colors ${isActive ? 'text-white' : 'text-gray-600 group-hover:text-gray-300'}`} 
      />
      <span className={`truncate text-left flex-1 ${isBold ? 'font-medium text-gray-200' : 'font-normal'}`}>
        {label}
      </span>
    </button>
  );
};

export default SidebarItem;