/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Restrained Near-Black & Charcoal Interface
        carbon: {
          DEFAULT: '#08090B', // Deep Black main background
          black: '#08090B',
          sidebar: '#0D0F12', // Sidebar & Header background
          deep: '#050607',
        },
        graphite: {
          DEFAULT: '#171A1F', // Main cards & panels
          surface: '#171A1F',
          elevated: '#20242A', // Elevated cards / modals
          hover: '#252A32',
        },
        // Restrained Official NVIDIA Green Palette
        nvidia: {
          light: '#8ED11A',   // NVIDIA Green light highlight
          highlight: '#8ED11A',
          DEFAULT: '#76B900', // Signature NVIDIA Green
          dark: '#5B8F00',    // Deeper NVIDIA Green
          dim: 'rgba(118, 185, 0, 0.12)',
          subtle: 'rgba(118, 185, 0, 0.05)',
        },
        // Legacy copper alias mapped to NVIDIA Green
        copper: {
          light: '#8ED11A',
          highlight: '#8ED11A',
          DEFAULT: '#76B900',
          dark: '#5B8F00',
          dim: 'rgba(118, 185, 0, 0.12)',
          subtle: 'rgba(118, 185, 0, 0.05)',
        },
        teal: {
          light: '#8CE0CF',
          DEFAULT: '#70D6C2', // Soft Teal secondary for non-blue telemetry
          dark: '#52BFA9',
          dim: 'rgba(112, 214, 194, 0.12)',
          subtle: 'rgba(112, 214, 194, 0.05)',
        },
        dark: {
          bg: '#08090B',         // Deep Black Main background
          surface: '#0D0F12',    // Sidebar & Top header background
          card: '#171A1F',       // Main cards (Graphite)
          elevated: '#20242A',   // Elevated panels / inner cards
          input: '#121519',      // Input backgrounds
          border: '#30353D',     // Hairline subtle borders
          cardHover: '#252A32',  // Hover background
          borderLight: '#3E444E',
        },
        text: {
          primary: '#F5F5F4',    // Soft White primary text
          secondary: '#A6ADB7',  // Refined Secondary text
          muted: '#7A828E',      // Muted helper text
        },
        status: {
          success: '#76B900',    // NVIDIA Green
          warning: '#FBBF24',
          error: '#FB7185',
        },
        // Functional telemetry accents
        telemetry: {
          green: '#76B900',
          teal: '#70D6C2',
          copper: '#76B900',
          amber: '#FBBF24',
          rose: '#FB7185',
        },
        obsidian: {
          800: '#252A32',
          850: '#171A1F',
          900: '#0D0F12',
          950: '#08090B',
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'Consolas', '"Courier New"', 'monospace'],
      },
    },
  },
  plugins: [],
}
