import type { Metadata } from 'next';
import { lato } from '@/lib/fonts';
import './globals.css';

export const metadata: Metadata = {
  title: 'RecommendIt - AI-Powered Recommendations',
  description:
    'Discover your next favorite books and movies with our hybrid recommendation system',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={lato.variable}>
      <body className="font-sans antialiased bg-secondary-50 text-secondary-900">
        {children}
      </body>
    </html>
  );
}
