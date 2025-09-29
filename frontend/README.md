# RecommendIt Frontend

A modern, responsive frontend for the RecommendIt hybrid recommendation system built with Next.js, TypeScript, and Tailwind CSS.

## Features

- 🎨 **Modern UI/UX** - Clean, professional design with Lato font and red theme (60-30-10 rule)
- 📱 **Responsive Design** - Works seamlessly on desktop, tablet, and mobile devices
- ⚡ **Fast Performance** - Built with Next.js 14 and optimized for speed
- 🔍 **Real-time Recommendations** - Get instant personalized recommendations
- 📚 **Multi-domain Support** - Switch between books and movies recommendations
- 📊 **Rich Data Display** - Beautiful cards showing recommendations with scores and metadata

## Design System

### Color Palette (60-30-10 Rule)

- **60% Primary (Red)**: `#ef4444` - Main brand color for CTAs and highlights
- **30% Secondary (Gray)**: `#6b7280` - Neutral colors for text and backgrounds
- **10% Accent (Blue)**: `#0ea5e9` - Accent color for special elements

### Typography

- **Font**: Lato (300, 400, 700, 900 weights)
- **Headings**: Bold, high contrast
- **Body**: Regular weight, optimal readability

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

1. Install dependencies:

```bash
npm install
```

2. Start the development server:

```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
npm run build
npm start
```

## Project Structure

```
src/
├── app/                    # Next.js app router
│   ├── about/             # About page
│   ├── api/               # API documentation page
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page
├── components/            # Reusable components
│   ├── DomainSelector.tsx # Books/Movies selector
│   ├── Header.tsx         # Navigation header
│   ├── LoadingSpinner.tsx # Loading indicator
│   └── RecommendationCard.tsx # Recommendation display
└── lib/                   # Utilities
    ├── fonts.ts           # Font configuration
    └── utils.ts           # Helper functions
```

## API Integration

The frontend connects to the backend API running on `http://localhost:8000`. Make sure the backend is running before using the frontend.

### Available Endpoints

- `POST /recommend` - Get personalized recommendations
- `GET /similar` - Find similar items
- `POST /feedback` - Submit user feedback
- `GET /healthz` - Health check

## Customization

### Colors

Update the color palette in `tailwind.config.ts`:

```typescript
colors: {
  primary: {
    // Red theme colors
  },
  secondary: {
    // Gray theme colors
  },
  accent: {
    // Blue accent colors
  }
}
```

### Fonts

Change the font in `src/lib/fonts.ts`:

```typescript
export const lato = Lato({
  subsets: ["latin"],
  weight: ["300", "400", "700", "900"],
  variable: "--font-lato",
});
```

## Deployment

### Vercel (Recommended)

1. Push to GitHub
2. Connect to Vercel
3. Deploy automatically

### Docker

```bash
docker build -t recommendit-frontend .
docker run -p 3000:3000 recommendit-frontend
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
