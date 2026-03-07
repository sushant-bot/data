import React from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Chip,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  CloudUpload as UploadIcon,
  TableChart as PreviewIcon,
  Tune as PreprocessIcon,
  Assessment as QualityIcon,
  ModelTraining as TrainIcon,
  BarChart as VizIcon,
  Psychology as AIIcon,
} from '@mui/icons-material';

const DRAWER_WIDTH = 260;

const navItems = [
  { label: 'Dashboard', path: '/', icon: <DashboardIcon /> },
  { label: 'Upload Data', path: '/upload', icon: <UploadIcon /> },
  { label: 'Data Preview', path: '/preview', icon: <PreviewIcon /> },
  { label: 'Preprocessing', path: '/preprocess', icon: <PreprocessIcon /> },
  { label: 'Quality Report', path: '/quality', icon: <QualityIcon /> },
  { label: 'ML Training', path: '/train', icon: <TrainIcon /> },
  { label: 'Visualizations', path: '/visualize', icon: <VizIcon /> },
  { label: 'AI Insights', path: '/recommendations', icon: <AIIcon /> },
];

interface LayoutProps {
  sessionId: string | null;
}

export default function Layout({ sessionId }: LayoutProps) {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: '#f5f7fa' }}>
      {/* Sidebar */}
      <Drawer
        variant="permanent"
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
            background: 'linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)',
            color: '#fff',
            borderRight: 'none',
          },
        }}
      >
        <Toolbar sx={{ px: 3, py: 2.5 }}>
          <Box>
            <Typography variant="h6" fontWeight={700} sx={{ letterSpacing: '-0.5px' }}>
              AI Data Analyst
            </Typography>
            <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.5)' }}>
              Intelligent Analytics Platform
            </Typography>
          </Box>
        </Toolbar>
        <Divider sx={{ borderColor: 'rgba(255,255,255,0.08)' }} />
        <List sx={{ px: 1.5, py: 1 }}>
          {navItems.map((item) => {
            const active = location.pathname === item.path;
            const needsSession = item.path !== '/' && item.path !== '/upload';
            const disabled = needsSession && !sessionId;

            return (
              <ListItemButton
                key={item.path}
                onClick={() => !disabled && navigate(item.path)}
                disabled={disabled}
                sx={{
                  borderRadius: 2,
                  mb: 0.5,
                  color: disabled ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.7)',
                  ...(active && {
                    bgcolor: 'rgba(99,102,241,0.2)',
                    color: '#818cf8',
                    '& .MuiListItemIcon-root': { color: '#818cf8' },
                  }),
                  '&:hover': {
                    bgcolor: active ? 'rgba(99,102,241,0.25)' : 'rgba(255,255,255,0.05)',
                  },
                }}
              >
                <ListItemIcon sx={{ color: 'inherit', minWidth: 40 }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.label}
                  primaryTypographyProps={{ fontSize: '0.875rem', fontWeight: active ? 600 : 400 }}
                />
              </ListItemButton>
            );
          })}
        </List>
      </Drawer>

      {/* Main content */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        <AppBar
          position="sticky"
          elevation={0}
          sx={{ bgcolor: '#fff', borderBottom: '1px solid #e5e7eb' }}
        >
          <Toolbar>
            <Typography variant="h6" sx={{ color: '#111827', fontWeight: 600, flexGrow: 1 }}>
              {navItems.find((n) => n.path === location.pathname)?.label || 'Dashboard'}
            </Typography>
            {sessionId && (
              <Chip
                label={`Session: ${sessionId.slice(0, 8)}...`}
                size="small"
                sx={{ bgcolor: '#eef2ff', color: '#4338ca', fontWeight: 500 }}
              />
            )}
          </Toolbar>
        </AppBar>

        <Box sx={{ p: 3, flexGrow: 1 }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
}
