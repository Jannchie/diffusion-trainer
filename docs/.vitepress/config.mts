import { defineConfig } from 'vitepress'

export default defineConfig({
  srcDir: 'zh',
  lang: 'zh-CN',
  title: 'Diffusion Trainer',
  description: '面向 SD 1.5 与 SDXL 的训练框架文档',
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ['meta', { name: 'theme-color', content: '#ee4c2c' }],
  ],
  themeConfig: {
    siteTitle: 'Diffusion Trainer',
    nav: [
      { text: '概览', link: '/' },
      { text: '快速开始', link: '/quick-start' },
      { text: '配置参考', link: '/configuration' },
      { text: '常见问题', link: '/faq' },
    ],
    sidebar: [
      {
        text: '开始使用',
        items: [
          { text: '项目简介', link: '/introduction' },
          { text: '安装', link: '/installation' },
          { text: '快速开始', link: '/quick-start' },
        ],
      },
      {
        text: '核心流程',
        items: [
          { text: '数据预处理', link: '/data-preparation' },
          { text: '训练流程', link: '/training' },
          { text: '数据集结构', link: '/dataset-format' },
        ],
      },
      {
        text: '参考',
        items: [
          { text: '配置说明', link: '/configuration' },
          { text: '开发说明', link: '/development' },
          { text: '常见问题', link: '/faq' },
        ],
      },
    ],
    outline: {
      level: [2, 3],
      label: '本页目录',
    },
    search: {
      provider: 'local',
    },
    footer: {
      message: 'Diffusion Trainer Documentation',
      copyright: 'Copyright © 2026',
    },
  },
})
