import React, { useEffect, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Skeleton,
  Alert,
  Grid,
  Card,
  CardContent,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Lightbulb as IdeaIcon,
  Warning as WarnIcon,
  TipsAndUpdates as TipIcon,
  ModelTraining as ModelIcon,
  AutoFixHigh as FeatureIcon,
  CheckCircle as QualityIcon,
} from '@mui/icons-material';
import { getRecommendations } from '../services/api';
import type { RecommendationsResponse } from '../types';

interface AIRecommendationsProps {
  sessionId: string;
}

export default function AIRecommendations({ sessionId }: AIRecommendationsProps) {
  const [data, setData] = useState<RecommendationsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getRecommendations(sessionId)
      .then((r) => { if (!cancelled) setData(r); })
      .catch((e) => { if (!cancelled) setError(e?.response?.data?.error || e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [sessionId]);

  if (loading) {
    return (
      <Box>
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} variant="rectangular" height={100} sx={{ borderRadius: 2, mb: 2 }} />
        ))}
      </Box>
    );
  }

  if (error) return <Alert severity="error" sx={{ borderRadius: 2 }}>{error}</Alert>;
  if (!data) return null;

  const { recommendations: recs, data_characteristics: chars } = data;

  const sections = [
    {
      title: 'Preprocessing Suggestions',
      icon: <TipIcon sx={{ color: '#6366f1' }} />,
      items: recs.preprocessing_suggestions,
      color: '#eef2ff',
    },
    {
      title: 'Model Suggestions',
      icon: <ModelIcon sx={{ color: '#0891b2' }} />,
      items: recs.model_suggestions,
      color: '#ecfeff',
    },
    {
      title: 'Feature Engineering Ideas',
      icon: <FeatureIcon sx={{ color: '#7c3aed' }} />,
      items: recs.feature_engineering_ideas,
      color: '#f5f3ff',
    },
    {
      title: 'Quality Recommendations',
      icon: <QualityIcon sx={{ color: '#22c55e' }} />,
      items: recs.quality_recommendations,
      color: '#f0fdf4',
    },
    {
      title: 'Warnings',
      icon: <WarnIcon sx={{ color: '#f59e0b' }} />,
      items: recs.warnings,
      color: '#fffbeb',
    },
  ];

  return (
    <Box>
      {/* Data characteristics overview */}
      <Paper elevation={0} sx={{ p: 2.5, mb: 3, border: '1px solid #e5e7eb', borderRadius: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="subtitle1" fontWeight={600}>Dataset Characteristics</Typography>
          {data.cached && <Chip label="Cached" size="small" variant="outlined" />}
        </Box>
        <Grid container spacing={2}>
          {[
            { label: 'Rows', value: chars.num_rows.toLocaleString() },
            { label: 'Columns', value: chars.num_columns },
            { label: 'Numeric', value: chars.num_numeric },
            { label: 'Categorical', value: chars.num_categorical },
            { label: 'Missing %', value: `${chars.missing_percentage.toFixed(1)}%` },
            { label: 'Duplicate %', value: `${chars.duplicate_percentage.toFixed(1)}%` },
          ].map((s) => (
            <Grid item xs={4} sm={2} key={s.label}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6" fontWeight={700}>{s.value}</Typography>
                <Typography variant="caption" color="text.secondary">{s.label}</Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Quality Score */}
      {recs.quality_score !== undefined && (
        <Paper elevation={0} sx={{ p: 2.5, mb: 3, border: '1px solid #e5e7eb', borderRadius: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
          <IdeaIcon sx={{ color: '#f59e0b', fontSize: 32 }} />
          <Box>
            <Typography variant="subtitle1" fontWeight={600}>AI Quality Score</Typography>
            <Typography variant="h4" fontWeight={800} sx={{ color: recs.quality_score >= 70 ? '#22c55e' : '#f59e0b' }}>
              {recs.quality_score.toFixed(0)}/100
            </Typography>
          </Box>
        </Paper>
      )}

      {/* Recommendation sections */}
      <Grid container spacing={2}>
        {sections
          .filter((s) => s.items.length > 0)
          .map((section) => (
            <Grid item xs={12} md={6} key={section.title}>
              <Paper
                elevation={0}
                sx={{ p: 2.5, border: '1px solid #e5e7eb', borderRadius: 2, height: '100%' }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                  {section.icon}
                  <Typography variant="subtitle2" fontWeight={600}>{section.title}</Typography>
                  <Chip label={section.items.length} size="small" sx={{ bgcolor: section.color, ml: 'auto' }} />
                </Box>
                <List dense disablePadding>
                  {section.items.map((item, i) => (
                    <ListItem key={i} disablePadding sx={{ py: 0.5 }}>
                      <ListItemText
                        primary={item}
                        primaryTypographyProps={{ variant: 'body2', sx: { lineHeight: 1.5 } }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            </Grid>
          ))}
      </Grid>
    </Box>
  );
}
