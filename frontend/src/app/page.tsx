'use client';

import { useState } from 'react';
import {
  Search,
  BookOpen,
  Film,
  Star,
  TrendingUp,
  Users,
  Zap,
} from 'lucide-react';
import Header from '@/components/Header';
import RecommendationCard from '@/components/RecommendationCard';
import DomainSelector from '@/components/DomainSelector';
import LoadingSpinner from '@/components/LoadingSpinner';

export default function Home() {
  const [selectedDomain, setSelectedDomain] = useState<'books' | 'movies'>(
    'books'
  );
  const [userId, setUserId] = useState('');
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGetRecommendations = async () => {
    if (!userId.trim()) {
      setError('Please enter a user ID');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`http://localhost:8000/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          domain: selectedDomain,
          k: 20,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }

      const data = await response.json();
      setRecommendations(data.recommendations || []);
    } catch (err) {
      setError('Failed to load recommendations. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const features = [
    {
      icon: <TrendingUp className="w-6 h-6" />,
      title: 'Hybrid AI',
      description:
        'Combines collaborative filtering and content-based recommendations',
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: 'Real-time',
      description: 'Get instant recommendations with sub-150ms response times',
    },
    {
      icon: <Users className="w-6 h-6" />,
      title: 'Personalized',
      description: 'Adapts to your preferences and interaction history',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-secondary-50 to-primary-50">
      <Header />

      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-secondary-900 mb-4">
            Discover Your Next
            <span className="text-primary-600"> Favorite</span>
          </h1>
          <p className="text-xl text-secondary-600 mb-8 max-w-2xl mx-auto">
            Get personalized recommendations for books and movies using our
            advanced hybrid AI system
          </p>

          {/* Search Section */}
          <div className="max-w-2xl mx-auto mb-8">
            <div className="flex flex-col sm:flex-row gap-4">
              <DomainSelector
                selectedDomain={selectedDomain}
                onDomainChange={setSelectedDomain}
              />
              <div className="flex-1">
                <input
                  type="text"
                  placeholder="Enter your user ID..."
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  className="input-field"
                />
              </div>
              <button
                onClick={handleGetRecommendations}
                disabled={loading}
                className="btn-primary flex items-center gap-2 whitespace-nowrap"
              >
                {loading ? <LoadingSpinner /> : <Search className="w-4 h-4" />}
                Get Recommendations
              </button>
            </div>
            {error && <p className="text-primary-600 mt-2 text-sm">{error}</p>}
          </div>
        </div>

        {/* Features Section */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {features.map((feature, index) => (
            <div key={index} className="card text-center">
              <div className="text-primary-600 mb-4 flex justify-center">
                {feature.icon}
              </div>
              <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
              <p className="text-secondary-600">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Recommendations Section */}
        {recommendations.length > 0 && (
          <div className="mb-12">
            <h2 className="text-3xl font-bold text-secondary-900 mb-6 text-center">
              Your {selectedDomain === 'books' ? 'Book' : 'Movie'}{' '}
              Recommendations
            </h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {recommendations.map((rec, index) => (
                <RecommendationCard
                  key={index}
                  recommendation={rec}
                  domain={selectedDomain}
                />
              ))}
            </div>
          </div>
        )}

        {/* Demo Section */}
        {recommendations.length === 0 && !loading && (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">
              {selectedDomain === 'books' ? (
                <BookOpen className="mx-auto text-primary-200" />
              ) : (
                <Film className="mx-auto text-primary-200" />
              )}
            </div>
            <h3 className="text-2xl font-semibold text-secondary-700 mb-2">
              Ready to discover?
            </h3>
            <p className="text-secondary-500">
              Enter your user ID above to get personalized recommendations
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
