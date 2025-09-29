'use client';

import { Star, Calendar, User, TrendingUp } from 'lucide-react';

interface RecommendationCardProps {
  recommendation: {
    item_id: string;
    title: string;
    score: number;
    source: string;
    author?: string;
    year?: number;
    genres?: string[];
    description?: string;
  };
  domain: 'books' | 'movies';
}

export default function RecommendationCard({
  recommendation,
  domain,
}: RecommendationCardProps) {
  const formatScore = (score: number) => {
    return (score * 100).toFixed(1);
  };

  const getSourceColor = (source: string) => {
    switch (source) {
      case 'als':
        return 'bg-accent-100 text-accent-700';
      case 'content':
        return 'bg-primary-100 text-primary-700';
      case 'item_item':
        return 'bg-secondary-100 text-secondary-700';
      default:
        return 'bg-secondary-100 text-secondary-700';
    }
  };

  const getSourceLabel = (source: string) => {
    switch (source) {
      case 'als':
        return 'Collaborative';
      case 'content':
        return 'Content-based';
      case 'item_item':
        return 'Similar Items';
      default:
        return source;
    }
  };

  return (
    <div className="card hover:shadow-md transition-shadow duration-200">
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1">
          <h3 className="font-semibold text-secondary-900 mb-1 line-clamp-2">
            {recommendation.title}
          </h3>
          {recommendation.author && (
            <p className="text-sm text-secondary-600 mb-2 flex items-center gap-1">
              <User className="w-3 h-3" />
              {recommendation.author}
            </p>
          )}
        </div>
        <div className="flex items-center gap-1 ml-2">
          <Star className="w-4 h-4 text-yellow-400 fill-current" />
          <span className="text-sm font-medium">
            {formatScore(recommendation.score)}%
          </span>
        </div>
      </div>

      {recommendation.description && (
        <p className="text-sm text-secondary-600 mb-3 line-clamp-3">
          {recommendation.description}
        </p>
      )}

      <div className="flex flex-wrap gap-2 mb-3">
        {recommendation.genres?.slice(0, 3).map((genre, index) => (
          <span
            key={index}
            className="px-2 py-1 bg-secondary-100 text-secondary-700 text-xs rounded-full"
          >
            {genre}
          </span>
        ))}
      </div>

      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span
            className={`px-2 py-1 text-xs rounded-full ${getSourceColor(
              recommendation.source
            )}`}
          >
            {getSourceLabel(recommendation.source)}
          </span>
          {recommendation.year && (
            <span className="text-xs text-secondary-500 flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              {recommendation.year}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 text-xs text-secondary-500">
          <TrendingUp className="w-3 h-3" />
          {recommendation.item_id}
        </div>
      </div>
    </div>
  );
}

