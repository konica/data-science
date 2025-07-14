<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";
</script>

<template>
  <header></header>

  <RouterView />
</template>

<style scoped>
header {
  line-height: 1.5;
  max-height: 100vh;
}

.logo {
  display: block;
  margin: 0 auto 2rem;
}
</style>
