import React from 'react';

/**
 * Top-Level Application Error Boundary
 * Prevents unhandled render/hook crashes from producing a blank screen.
 */
export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('SignBridge ErrorBoundary caught an error:', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: '100vh',
          backgroundColor: '#1E1B18',
          color: '#F2EAE0',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem',
          fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
          textAlign: 'center'
        }}>
          <div style={{
            maxWidth: '520px',
            backgroundColor: '#2A2622',
            padding: '2.5rem',
            borderRadius: '16px',
            border: '1px solid rgba(201, 168, 131, 0.25)',
            boxShadow: '0 20px 40px rgba(0,0,0,0.5)'
          }}>
            <div style={{
              width: '56px',
              height: '56px',
              borderRadius: '50%',
              backgroundColor: 'rgba(116, 128, 106, 0.2)',
              color: '#74806A',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1.5rem',
              fontSize: '1.75rem'
            }}>
              ⚠️
            </div>
            <h2 style={{ margin: '0 0 0.75rem', fontSize: '1.5rem', fontWeight: 700 }}>
              SignBridge Encountered an Issue
            </h2>
            <p style={{ margin: '0 0 1.5rem', color: '#C9A883', fontSize: '0.95rem', lineHeight: 1.5 }}>
              The application recovered from an unexpected error. You can refresh the kiosk to resume camera tracking and translation.
            </p>
            {this.state.error?.message && (
              <pre style={{
                textAlign: 'left',
                backgroundColor: 'rgba(0,0,0,0.3)',
                padding: '0.75rem 1rem',
                borderRadius: '8px',
                fontSize: '0.8rem',
                color: '#E07A5F',
                overflowX: 'auto',
                marginBottom: '1.5rem'
              }}>
                {this.state.error.message}
              </pre>
            )}
            <button
              onClick={this.handleRetry}
              style={{
                backgroundColor: '#74806A',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '8px',
                padding: '0.75rem 1.75rem',
                fontSize: '1rem',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'background-color 0.2s ease'
              }}
              onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#626E59'}
              onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#74806A'}
            >
              Restart Kiosk
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
