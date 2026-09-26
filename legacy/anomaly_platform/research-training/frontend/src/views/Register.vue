<template>
  <div class="register-container">
    <el-card class="register-card" :header="$t('创建账号（Create Account）')">
      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        @submit.prevent="handleRegister"
        label-position="top"
      >
        <el-form-item :label="$t('用户名（Username）')" prop="username">
          <el-input v-model="form.username" :placeholder="$t('输入用户名（Enter username）')" />
        </el-form-item>
        <el-form-item :label="$t('密码（Password）')" prop="password">
          <el-input v-model="form.password" type="password" :placeholder="$t('至少 6 位（At least 6 characters）')" show-password />
        </el-form-item>
        <el-form-item :label="$t('确认密码（Confirm Password）')" prop="confirmPassword">
          <el-input v-model="form.confirmPassword" type="password" :placeholder="$t('再次输入密码（Confirm password）')" show-password />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" native-type="submit" :loading="loading" style="width:100%">
            {{ $t('注册（Register）') }}
          </el-button>
        </el-form-item>
        <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" />
        <div style="text-align:center; margin-top:12px">
          <span style="color:#666; font-size:13px;">
            {{ $t('已有账号？（Already have an account?）') }}
            <router-link to="/login" style="color:#409eff; text-decoration:none;">{{ $t('去登录（Log in）') }}</router-link>
          </span>
        </div>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { register as registerApi } from '../api'

const router = useRouter()
const formRef = ref(null)
const loading = ref(false)
const error = ref('')

const form = reactive({
  username: '',
  password: '',
  confirmPassword: '',
})

const validateConfirm = (rule, value, callback) => {
  if (value !== form.password) {
    callback(new Error('两次密码不一致（Passwords do not match）'))
  } else {
    callback()
  }
}

const rules = {
  username: [
    { required: true, message: '用户名不能为空（Username is required）', trigger: 'blur' },
  ],
  password: [
    { required: true, message: '密码不能为空（Password is required）', trigger: 'blur' },
    { min: 6, message: '密码至少 6 位（At least 6 characters）', trigger: 'blur' },
  ],
  confirmPassword: [
    { required: true, message: '请确认密码（Please confirm your password）', trigger: 'blur' },
    { validator: validateConfirm, trigger: 'blur' },
  ],
}

async function handleRegister() {
  if (!formRef.value) return
  const valid = await formRef.value.validate().catch(() => false)
  if (!valid) return

  loading.value = true
  error.value = ''
  try {
    await registerApi(form.username, form.password, form.confirmPassword)
    ElMessage.success('注册成功！请登录。（Registration successful! Please log in.）')
    router.push('/login')
  } catch (e) {
    error.value = e.response?.data?.detail || '注册失败（Registration failed）'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.register-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.register-card {
  width: 380px;
}
</style>
