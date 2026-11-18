export type SectionTab =
  | 'showcase'
  | 'overview'
  | 'gpu_monitoring'
  | 'training_studio'
  | 'model_registry'
  | 'cuda_lab'
  | 'analytics'
  | 'inference'
  | 'verification';

export interface NavItem {
  id: SectionTab;
  label: string;
  badge?: string;
  shortcut: string;
  category: 'core' | 'engineering' | 'models' | 'audit';
}
