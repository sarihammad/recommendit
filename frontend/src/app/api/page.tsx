'use client'

import { Code, Play, Copy, Check } from 'lucide-react'
import { useState } from 'react'
import Header from '@/components/Header'

export default function API() {
  const [copiedEndpoint, setCopiedEndpoint] = useState<string | null>(null)

  const copyToClipboard = (text: string, endpoint: string) => {
    navigator.clipboard.writeText(text)
    setCopiedEndpoint(endpoint)
    setTimeout(() => setCopiedEndpoint(null), 2000)
  }

  const endpoints = [
    {
      method: 'POST',
      path: '/recommend',
      description: 'Get personalized recommendations for a user',
      example: {
        request: {
          user_id: "user123",
          domain: "books",
          k: 20
        },
        response: {
          recommendations: [
            {
              item_id: "book456",
              title: "The Great Adventure",
              score: 0.95,
              source: "als",
              author: "John Smith",
              year: 2020,
              genres: ["Adventure", "Fiction"]
            }
          ]
        }
      }
    },
    {
      method: 'GET',
      path: '/similar',
      description: 'Find similar items to a given item',
      example: {
        request: {
          item_id: "book456",
          domain: "books",
          k: 10
        },
        response: {
          similar_items: [
            {
              item_id: "book789",
              title: "Another Adventure",
              score: 0.87,
              source: "content"
            }
          ]
        }
      }
    },
    {
      method: 'POST',
      path: '/feedback',
      description: 'Submit user feedback for model improvement',
      example: {
        request: {
          user_id: "user123",
          item_id: "book456",
          event_type: "click",
          timestamp: "2024-01-15T10:30:00Z"
        },
        response: {
          status: "success",
          message: "Feedback recorded"
        }
      }
    },
    {
      method: 'GET',
      path: '/healthz',
      description: 'Check API health status',
      example: {
        response: {
          status: "healthy",
          timestamp: "2024-01-15T10:30:00Z",
          version: "1.0.0"
        }
      }
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-secondary-50 to-primary-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-secondary-900 mb-4">
            API <span className="text-primary-600">Documentation</span>
          </h1>
          <p className="text-xl text-secondary-600 max-w-3xl mx-auto">
            Integrate with our recommendation system using our RESTful API. 
            All endpoints return JSON responses and support CORS.
          </p>
        </div>

        {/* Base URL */}
        <div className="card mb-8">
          <h2 className="text-2xl font-bold text-secondary-900 mb-4">Base URL</h2>
          <div className="bg-secondary-100 p-4 rounded-lg flex items-center justify-between">
            <code className="text-lg font-mono">http://localhost:8000</code>
            <button
              onClick={() => copyToClipboard('http://localhost:8000', 'base')}
              className="flex items-center gap-2 text-primary-600 hover:text-primary-700"
            >
              {copiedEndpoint === 'base' ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              {copiedEndpoint === 'base' ? 'Copied!' : 'Copy'}
            </button>
          </div>
        </div>

        {/* Endpoints */}
        <div className="space-y-8">
          {endpoints.map((endpoint, index) => (
            <div key={index} className="card">
              <div className="flex items-center gap-4 mb-4">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  endpoint.method === 'GET' 
                    ? 'bg-accent-100 text-accent-700' 
                    : 'bg-primary-100 text-primary-700'
                }`}>
                  {endpoint.method}
                </span>
                <code className="text-lg font-mono bg-secondary-100 px-3 py-1 rounded">
                  {endpoint.path}
                </code>
                <button
                  onClick={() => copyToClipboard(endpoint.path, `endpoint-${index}`)}
                  className="flex items-center gap-1 text-secondary-600 hover:text-secondary-900"
                >
                  {copiedEndpoint === `endpoint-${index}` ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </button>
              </div>
              
              <p className="text-secondary-600 mb-6">{endpoint.description}</p>

              {/* Request Example */}
              {endpoint.example.request && (
                <div className="mb-6">
                  <h4 className="text-lg font-semibold mb-3 flex items-center gap-2">
                    <Play className="w-4 h-4" />
                    Request Example
                  </h4>
                  <div className="bg-secondary-900 text-secondary-100 p-4 rounded-lg overflow-x-auto">
                    <pre className="text-sm">
                      {JSON.stringify(endpoint.example.request, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {/* Response Example */}
              <div>
                <h4 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Code className="w-4 h-4" />
                  Response Example
                </h4>
                <div className="bg-secondary-900 text-secondary-100 p-4 rounded-lg overflow-x-auto">
                  <pre className="text-sm">
                    {JSON.stringify(endpoint.example.response, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Authentication */}
        <div className="card mt-12">
          <h2 className="text-2xl font-bold text-secondary-900 mb-4">Authentication</h2>
          <p className="text-secondary-600 mb-4">
            Currently, the API is open and doesn't require authentication. 
            In production, you would typically use API keys or OAuth tokens.
          </p>
          <div className="bg-secondary-100 p-4 rounded-lg">
            <code className="text-sm">
              # Example with API key (future implementation)<br/>
              curl -H "Authorization: Bearer YOUR_API_KEY" \<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;-H "Content-Type: application/json" \<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;-d '{"user_id": "user123", "domain": "books", "k": 20}' \<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;http://localhost:8000/recommend
            </code>
          </div>
        </div>

        {/* Rate Limits */}
        <div className="card mt-8">
          <h2 className="text-2xl font-bold text-secondary-900 mb-4">Rate Limits</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">Current Limits</h3>
              <ul className="text-secondary-600 space-y-1">
                <li>• 1000 requests per minute</li>
                <li>• 10,000 requests per hour</li>
                <li>• No authentication required</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-2">Response Headers</h3>
              <ul className="text-secondary-600 space-y-1">
                <li>• <code>X-RateLimit-Limit</code> - Request limit</li>
                <li>• <code>X-RateLimit-Remaining</code> - Remaining requests</li>
                <li>• <code>X-RateLimit-Reset</code> - Reset timestamp</li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}



