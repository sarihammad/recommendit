'use client';

import { BookOpen, Film } from 'lucide-react';

interface DomainSelectorProps {
  selectedDomain: 'books' | 'movies';
  onDomainChange: (domain: 'books' | 'movies') => void;
}

export default function DomainSelector({
  selectedDomain,
  onDomainChange,
}: DomainSelectorProps) {
  return (
    <div className="flex bg-secondary-100 rounded-lg p-1">
      <button
        onClick={() => onDomainChange('books')}
        className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all duration-200 ${
          selectedDomain === 'books'
            ? 'bg-white text-primary-600 shadow-sm'
            : 'text-secondary-600 hover:text-secondary-900'
        }`}
      >
        <BookOpen className="w-4 h-4" />
        Books
      </button>
      <button
        onClick={() => onDomainChange('movies')}
        className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all duration-200 ${
          selectedDomain === 'movies'
            ? 'bg-white text-primary-600 shadow-sm'
            : 'text-secondary-600 hover:text-secondary-900'
        }`}
      >
        <Film className="w-4 h-4" />
        Movies
      </button>
    </div>
  );
}
