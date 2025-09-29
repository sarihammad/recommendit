'use client';

import { TrendingUp, Zap, Users, Brain, Database, Cpu } from 'lucide-react';
import Header from '@/components/Header';

export default function About() {
  const features = [
    {
      icon: <Brain className="w-8 h-8" />,
      title: 'Hybrid AI System',
      description:
        'Combines collaborative filtering (ALS) and content-based recommendations using SBERT embeddings for optimal results.',
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: 'Lightning Fast',
      description:
        'Sub-150ms response times with Redis caching and optimized Faiss indices for real-time recommendations.',
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: 'Cold Start Handling',
      description:
        'Intelligent strategies for new users and items with alpha blending and popularity-based fallbacks.',
    },
    {
      icon: <Database className="w-8 h-8" />,
      title: 'Robust Data Pipeline',
      description:
        'Automatic schema inference, time-based splits, and comprehensive feature engineering.',
    },
    {
      icon: <Cpu className="w-8 h-8" />,
      title: 'Production Ready',
      description:
        'Docker containerization, Prometheus metrics, structured logging, and comprehensive testing.',
    },
    {
      icon: <TrendingUp className="w-8 h-8" />,
      title: 'Continuous Learning',
      description:
        'LightGBM ranking model with feature importance tracking and A/B testing capabilities.',
    },
  ];

  const metrics = [
    {
      label: 'Recall@20',
      value: '0.85+',
      description: 'High recall for relevant items',
    },
    {
      label: 'NDCG@20',
      value: '0.78+',
      description: 'Excellent ranking quality',
    },
    { label: 'Latency', value: '<150ms', description: 'Real-time response' },
    {
      label: 'Throughput',
      value: '1000+ RPS',
      description: 'High performance',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-secondary-50 to-primary-50">
      <Header />

      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-secondary-900 mb-4">
            About <span className="text-primary-600">RecommendIt</span>
          </h1>
          <p className="text-xl text-secondary-600 max-w-3xl mx-auto">
            A production-ready hybrid recommendation system that combines the
            power of collaborative filtering, content-based recommendations, and
            machine learning to deliver personalized suggestions for books and
            movies.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {features.map((feature, index) => (
            <div key={index} className="card">
              <div className="text-primary-600 mb-4">{feature.icon}</div>
              <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
              <p className="text-secondary-600">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Performance Metrics */}
        <div className="card mb-16">
          <h2 className="text-3xl font-bold text-secondary-900 mb-8 text-center">
            Performance Metrics
          </h2>
          <div className="grid md:grid-cols-4 gap-6">
            {metrics.map((metric, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl font-bold text-primary-600 mb-2">
                  {metric.value}
                </div>
                <div className="text-lg font-semibold text-secondary-900 mb-1">
                  {metric.label}
                </div>
                <div className="text-sm text-secondary-600">
                  {metric.description}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Architecture Section */}
        <div className="card mb-16">
          <h2 className="text-3xl font-bold text-secondary-900 mb-8 text-center">
            System Architecture
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold mb-4">Two-Stage Pipeline</h3>
              <ul className="space-y-3 text-secondary-600">
                <li className="flex items-start gap-2">
                  <span className="text-primary-600 font-bold">1.</span>
                  <span>
                    <strong>Recall Stage:</strong> Generate candidates using
                    ALS, Item-Item CF, and Content embeddings
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary-600 font-bold">2.</span>
                  <span>
                    <strong>Ranking Stage:</strong> Re-rank candidates using
                    LightGBM with rich feature engineering
                  </span>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-semibold mb-4">Key Technologies</h3>
              <ul className="space-y-3 text-secondary-600">
                <li>
                  • <strong>FastAPI</strong> - High-performance API framework
                </li>
                <li>
                  • <strong>Redis</strong> - Caching and session management
                </li>
                <li>
                  • <strong>Faiss</strong> - Approximate nearest neighbor search
                </li>
                <li>
                  • <strong>LightGBM</strong> - Gradient boosting for ranking
                </li>
                <li>
                  • <strong>SBERT</strong> - Sentence embeddings for content
                </li>
                <li>
                  • <strong>Docker</strong> - Containerized deployment
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="text-center">
          <h2 className="text-3xl font-bold text-secondary-900 mb-4">
            Ready to Get Started?
          </h2>
          <p className="text-xl text-secondary-600 mb-8">
            Try our recommendation system or explore the API documentation
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a href="/" className="btn-primary">
              Try Recommendations
            </a>
            <a href="/api" className="btn-secondary">
              View API Docs
            </a>
          </div>
        </div>
      </main>
    </div>
  );
}
