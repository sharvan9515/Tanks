import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Tank Cost Estimator",
  description: "Interactive demo for the tank cost prediction model"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-100 antialiased">{children}</body>
    </html>
  );
}
