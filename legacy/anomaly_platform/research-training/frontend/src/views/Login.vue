<template>
  <div class="login-container">
    <el-card class="login-card" :header="$t('FedTAD Training Manager')">
      <el-form @submit.prevent="handleLogin" label-position="top">
        <el-form-item :label="$t('用户名（Username）')">
          <el-input v-model="username" :placeholder="$t('输入用户名（Enter username）')" />
        </el-form-item>
        <el-form-item :label="$t('密码（Password）')">
          <el-input v-model="password" type="password" :placeholder="$t('输入密码（Enter password）')" show-password />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" native-type="submit" :loading="loading" style="width:100%">
            {{ $t('登录（Login）') }}
          </el-button>
        </el-form-item>
        <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" />
        <div style="text-align:center; margin-top:12px">
          <span style="color:#666; font-size:13px;">
            {{ $t('没有账号？（No account?）') }}
            <router-link to="/register" style="color:#409eff; text-decoration:none;">{{ $t('去注册（Register）') }}</router-link>
          </span>
        </div>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const auth = useAuthStore()

const username = ref('admin')
const password = ref('admin123')
const loading = ref(false)
const error = ref('')

async function handleLogin() {
  loading.value = true
  error.value = ''
  try {
    await auth.login(username.value, password.value)
    router.push('/')
  } catch (e) {
    error.value = e.response?.data?.detail || '登录失败（Login failed）'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.login-card {
  width: 380px;
}
</style>
