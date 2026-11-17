// workgroup sum / max for any subgroup size (red_sh must hold one slot per subgroup)
shared float red_sh[64];
float wg_sum(float v) {
  v = subgroupAdd(v);
  if (subgroupElect()) red_sh[gl_SubgroupID] = v;
  barrier();
  float s = 0.0;
  for (uint i = 0u; i < gl_NumSubgroups; i++) s += red_sh[i];
  barrier();
  return s;
}
