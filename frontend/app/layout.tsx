import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
//@ts-expect-error : ignore
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Lethus AI",
  description:
    "Overcome the limitations of large language models with intelligent context management. Our system prioritizes critical information, reduces noise, prevents hallucinations, and optimizes token usage for faster, more accurate AI interactions.",
  icons: {
    icon: "/lethusai.png",
    shortcut: "/lethusai.png",
    apple: "/lethusai.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
