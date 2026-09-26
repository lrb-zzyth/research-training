import { createRouter, createWebHistory } from 'vue-router'
import Login from '../views/Login.vue'
import Register from '../views/Register.vue'
import Dashboard from '../views/Dashboard.vue'
import TrainingConfig from '../views/TrainingConfig.vue'
import TrainingMonitor from '../views/TrainingMonitor.vue'
import ExperimentHistory from '../views/ExperimentHistory.vue'
import Settings from '../views/Settings.vue'

const publicRoutes = ['Login', 'Register']

const routes = [
  { path: '/login', name: 'Login', component: Login },
  { path: '/register', name: 'Register', component: Register },
  {
    path: '/',
    component: Dashboard,
    redirect: '/training/config',
    children: [
      { path: 'training/config', name: 'TrainingConfig', component: TrainingConfig },
      { path: 'training/monitor/:id?', name: 'TrainingMonitor', component: TrainingMonitor },
      { path: 'experiments', name: 'ExperimentHistory', component: ExperimentHistory },
      { path: 'settings', name: 'Settings', component: Settings },
    ],
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  if (!token && !publicRoutes.includes(to.name)) {
    next('/login')
  } else {
    next()
  }
})

export default router
