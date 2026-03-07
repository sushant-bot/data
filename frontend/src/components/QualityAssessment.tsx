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
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  CheckCircle as GoodIcon,
  Warning as WarnIcon,
  Error as BadIcon,
  Lightbulb as TipIcon,
} from '@mui/icons-material';
import { getQuality } from '../services/api';
import type { QualityResponse } from '../types';

interface QualityProps {
  sessionId: string;
}

export default function QualityAssessment({ sessionId }: QualityProps) {
  const [data, setData] = useState<QualityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getQuality(sessionId)
      .then((r) => { if (!cancelled) setData(r); })
      .catch((e) => { if (!cancelled) setError(e?.response?.data?.error || e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [sessionId]);

  if (loading) {
    return (
      <Box>
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} variant="rectangular" height={80} sx={{ borderRadius: 2, mb: 2 }} />
        ))}
      </Box>
    );
  }

  if (error) return <Alert severity="error" sx={{ borderRadius: 2 }}>{error}</Alert>;
  if (!data) return null;

  const q = data.quality_report;
  const score = q.overall_quality_score;
  const scoreColor = score >= 80 ? '#22c55e' : score >= 60 ? '#f59e0b' : '#ef4444';

  return (
    <Box>
      {/* Quality Score */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          mb: 3,
          border: '1px solid #e5e7eb',
          borderRadius: 3,
          display: 'flex',
          alignItems: 'center',
          gap: 3,
        }}
      >
        <Box
          sx={{
            width: 80,
            height: 80,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: `4px solid ${scoreColor}`,
          }}
        >
          <Typography variant="h4" fontWeight={800} sx={{ color: scoreColor }}>
            {Math.round(score)}
          </Typography>
        </Box>
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="h6" fontWeight={600}>Overall Quality Score</Typography>
          <LinearProgress
            variant="determinate"
            value={score}
            sx={{
              height: 8,
              borderRadius: 4,
              mt: 1,
              bgcolor: '#f3f4f6',
              '& .MuiLinearProgress-bar': { bgcolor: scoreColor, borderRadius: 4 },
            }}
          />
        </Box>
        <Chip
          label={data.dataset_type}
          size="small"
          sx={{ bgcolor: '#eef2ff', color: '#4338ca', textTransform: 'capitalize' }}
        />
      </Paper>

      {/* Metrics Grid */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} md={3}>
          <MetricCard label="Total Rows" value={q.basic_metrics.total_rows.toLocaleString()} />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard label="Total Columns" value={q.basic_metrics.total_columns} />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard label="Memory" value={`${q.basic_metrics.memory_usage_mb.toFixed(1)} MB`} />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard label="Total Cells" value={q.basic_metrics.total_cells.toLocaleString()} />
        </Grid>
      </Grid>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        {/* Missing Values */}
        <Grid item xs={12} md={6}>
          <Paper elevation={0} sx={{ p: 2.5, border: '1px solid #e5e7eb', borderRadius: 2, height: '100%' }}>
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>Missing Values</Typography>
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <Chip
                label={`${q.missing_value_analysis.total_missing_values} total`}
                size="small"
                color={q.missing_value_analysis.total_missing_values > 0 ? 'warning' : 'success'}
              />
              <Chip
                label={`${q.missing_value_analysis.total_missing_percentage.toFixed(1)}%`}
                size="small"
                variant="outlined"
              />
            </Box>
            {q.missing_value_analysis.high_missing_columns.length > 0 && (
              <Alert severity="warning" sx={{ mb: 1, py: 0 }} icon={<WarnIcon fontSize="small" />}>
                High: {q.missing_value_analysis.high_missing_columns.join(', ')}
              </Alert>
            )}
            {q.missing_value_analysis.complete_columns.length > 0 && (
              <Typography variant="body2" color="text.secondary">
                {q.missing_value_analysis.complete_columns.length} column(s) fully complete
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Duplicates */}
        <Grid item xs={12} md={6}>
          <Paper elevation={0} sx={{ p: 2.5, border: '1px solid #e5e7eb', borderRadius: 2, height: '100%' }}>
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>Duplicates</Typography>
            <Box sx={{ display: 'flex', gap: 2, mb: 1 }}>
              <Chip
                label={`${q.duplicate_analysis.duplicate_rows} duplicate rows`}
                size="small"
                color={q.duplicate_analysis.has_duplicates ? 'warning' : 'success'}
              />
              <Chip
                label={`${q.duplicate_analysis.duplicate_percentage.toFixed(1)}%`}
                size="small"
                variant="outlined"
              />
            </Box>
            <Typography variant="body2" color="text.secondary">
              {q.duplicate_analysis.unique_rows.toLocaleString()} unique rows
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Data Types */}
      <Paper elevation={0} sx={{ p: 2.5, mb: 3, border: '1px solid #e5e7eb', borderRadius: 2 }}>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>Data Types</Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {Object.entries(q.data_type_analysis.type_distribution).map(([type, count]) => (
            <Chip key={type} label={`${type}: ${count}`} variant="outlined" size="small" />
          ))}
        </Box>
        {q.data_type_analysis.potential_conversions.length > 0 && (
          <Alert severity="info" sx={{ mt: 2, py: 0 }}>
            Potential conversions: {q.data_type_analysis.potential_conversions.join(', ')}
          </Alert>
        )}
      </Paper>

      {/* Recommendations */}
      {q.recommendations.length > 0 && (
        <Paper elevation={0} sx={{ p: 2.5, border: '1px solid #e5e7eb', borderRadius: 2 }}>
          <Typography variant="subtitle1" fontWeight={600} gutterBottom>Recommendations</Typography>
          <List dense disablePadding>
            {q.recommendations.map((rec, i) => (
              <ListItem key={i} disablePadding sx={{ py: 0.5 }}>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <TipIcon fontSize="small" sx={{ color: '#f59e0b' }} />
                </ListItemIcon>
                <ListItemText primary={rec} primaryTypographyProps={{ variant: 'body2' }} />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}
    </Box>
  );
}

function MetricCard({ label, value }: { label: string; value: any }) {
  return (
    <Card elevation={0} sx={{ border: '1px solid #e5e7eb', borderRadius: 2 }}>
      <CardContent sx={{ textAlign: 'center', py: 2 }}>
        <Typography variant="h5" fontWeight={700}>{value}</Typography>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
      </CardContent>
    </Card>
  );
}
