<template>
  Welcome to the temperature prediction app! This app uses a linear regression
  model to predict the temperature in London based on historical data. You need
  to provide the following inputs: 'min_temp', 'global_radiation', 'month',
  'cloud_cover'
  <div>Chat:</div>
  <div>
    <textarea
      v-model="chatInput"
      placeholder="Type your message here..."
      rows="4"
    ></textarea>
  </div>
</template>
<script setup lang="ts"></script>
