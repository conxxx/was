import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom'; // Import Link
import apiService from '../services/api';
import InputField from '../components/fields/InputField'; // Import the new InputField
import authImg from '../assets/img/auth/auth.png'; // Import the background image

function LoginPage() {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  // Redirect if already logged in (Keep existing logic)
  useEffect(() => {
    const clientId = localStorage.getItem('clientId');
    if (clientId) {
      navigate('/dashboard', { replace: true });
    }
    // Also update the initial redirect check if already logged in
    if (clientId) {
      navigate('/admin/dashboard', { replace: true }); 
    }
  }, [navigate]);

  // Keep existing login logic
  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    if (!email) {
      setError('Please enter your email.');
      return;
    }
    setIsLoading(true);
    try {
      await apiService.login(email);
      // Update redirect on successful login
      navigate('/admin/dashboard'); 
    } catch (err) {
      setError(err.message || 'Login failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Apply Horizon-based structure and styling - Simplified
  return (
    <div className="flex min-h-screen w-full !bg-white dark:!bg-navy-900"> {/* Main flex container */}

      {/* Left Column (Form Area) */}
      <div className="flex flex-1 flex-col justify-center p-4 lg:pl-[70px] lg:pr-4"> {/* Changed items-center to flex-col justify-center */}
        <div className="w-full max-w-[420px] self-center"> {/* Added self-center */}
          <h4 className="mb-2.5 text-4xl font-bold text-navy-700 dark:text-white">
            Sign In {/* Updated Text */}
          </h4>
          <p className="mb-9 ml-1 text-base text-gray-600">
            Enter your email and password to sign in! {/* Updated Text */}
          </p>

          {error && <p className="mb-4 text-sm text-red-500">{error}</p>}

          <form onSubmit={handleLogin}>
            <InputField
              variant="auth"
              extra="mb-3"
              label="Email*"
              placeholder="mail@example.com"
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              state={error ? "error" : ""}
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading}
              className="linear mt-2 w-full rounded-xl bg-brand-500 py-[12px] text-base font-medium text-white transition duration-200 hover:bg-brand-600 active:bg-brand-700 dark:bg-brand-400 dark:text-white dark:hover:bg-brand-300 dark:active:bg-brand-200 disabled:opacity-50"
            >
              {isLoading ? 'Processing...' : 'Sign In'} {/* Updated Text */}
            </button>
            {/* Add Home Link */}
            <div className="mt-4 flex justify-center">
              <Link to="/" className="text-sm font-medium text-brand-500 hover:text-brand-600 dark:text-white">
                Back to Home
              </Link>
            </div>
          </form>
        </div>
        {/* Footer */}
        <p className="pt-10 text-sm text-gray-600 text-center">
          ©{new Date().getFullYear()} Horizon UI. All Rights Reserved.
        </p>
      </div>

      {/* Right Column (Image Area) */}
      <div className="relative hidden h-screen md:flex md:flex-col md:items-center md:justify-center lg:w-[49vw] 2xl:w-[44vw]"> {/* Fixed width column, hidden on small screens, added flex centering for overlay */}
        <div
          className="absolute inset-0 bg-cover bg-center lg:rounded-bl-[120px] xl:rounded-bl-[200px]"
          style={{ backgroundImage: `url(${authImg})` }}
        />
        {/* Overlay Content */}
        <div className="relative z-10 flex flex-col items-center text-white">
          {/* Placeholder for Logo */}
          <div className="mb-4 h-16 w-16 rounded-full bg-white flex items-center justify-center text-navy-700 font-bold text-xl">
            HL {/* Logo Placeholder */}
          </div>
          <h2 className="mb-4 text-3xl font-bold">Horizon UI</h2>
          <a
            href="https://horizon-ui.com"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-lg border border-white px-6 py-3 text-sm font-medium transition duration-200 hover:bg-white hover:text-navy-700"
          >
            Learn more about Horizon UI on horizon-ui.com
          </a>
        </div>
      </div>

    </div>
  );
}

export default LoginPage;
