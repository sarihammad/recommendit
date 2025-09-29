'use client';

import { BookOpen, Film, Star } from 'lucide-react';

export default function Header() {
  return (
    <header className="bg-white shadow-sm border-b border-secondary-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-primary-600 p-2 rounded-lg">
              <Star className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-secondary-900">
              RecommendIt
            </h1>
          </div>

          <nav className="hidden md:flex items-center gap-6">
            <a
              href="#"
              className="text-secondary-600 hover:text-primary-600 transition-colors"
            >
              Home
            </a>
            <a
              href="#"
              className="text-secondary-600 hover:text-primary-600 transition-colors"
            >
              About
            </a>
            <a
              href="#"
              className="text-secondary-600 hover:text-primary-600 transition-colors"
            >
              API
            </a>
          </nav>

          <div className="flex items-center gap-2">
            <div className="bg-secondary-100 p-2 rounded-lg">
              <BookOpen className="w-4 h-4 text-secondary-600" />
            </div>
            <div className="bg-secondary-100 p-2 rounded-lg">
              <Film className="w-4 h-4 text-secondary-600" />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

